# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict

import aiofiles
import dotenv
import openai
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

# from summary_time_cost import generate_summary
from customized_areal.tpfc.backend_run import run_backend
from customized_areal.tpfc.eval_utils import verify_answer_for_datasets

dotenv.load_dotenv()


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUN_FAILED = "run_failed"
    RUN_COMPLETED = "run_completed"
    RESULT_JUDGED = "result_judged"


@dataclass
class BenchmarkTask:
    """Generic benchmark task data structure"""

    task_id: str
    task_question: str
    ground_truth: str
    file_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    model_response: str = ""
    model_boxed_answer: str = ""
    status: TaskStatus = TaskStatus.PENDING
    Level: str = ""
    # status: str = "pending"  # pending, success, failed


class AttemptStats(TypedDict):
    attempt_number: int
    model_response: str
    model_boxed_answer: str
    status: TaskStatus
    log_file_path: Path | None
    llm_as_judge_result: str | None
    is_correct: bool
    error_message: str | None


class BenchmarkResult(BaseModel):
    """Generic benchmark evaluation result structure"""

    task_id: str
    task_question: str
    ground_truth: str
    file_path: str | None
    model_response: str
    model_boxed_answer: str
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""
    llm_as_judge_result: str | None = None
    log_file_path: Path | None = None
    # Pass@K support fields
    attempts: list[AttemptStats] = Field(default_factory=list)  # Store all attempts
    pass_at_k_success: bool = False  # Whether task passed using pass@k evaluation
    k_value: int = 1  # The k value used for this evaluation


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            _message_content_to_text(part)
            if isinstance(part, (dict, list))
            else str(part)
            for part in content
        )
    if isinstance(content, dict):
        value = content.get("text", content.get("content", ""))
        return _message_content_to_text(value)
    return str(content) if content is not None else ""


def _last_assistant_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _message_content_to_text(message.get("content", ""))
    return ""


class BenchmarkEvaluator(ABC):
    """Abstract base class for benchmark evaluators"""

    def __init__(self, data_dir: str, benchmark_name: str, cfg: DictConfig):
        """
        Initialize benchmark evaluator

        Args:
            data_dir: Path to benchmark data directory
            benchmark_name: Name of the benchmark
            cfg: The Hydra configuration object
        """
        self.data_dir = Path(data_dir)
        self.benchmark_name = benchmark_name
        self.cfg = cfg
        self.pass_at_k = cfg.benchmark.execution.get("pass_at_k", 1)
        self.output_dir = Path(cfg.output_dir).absolute()
        if not self.output_dir.exists():
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Created output directory: {self.output_dir}")
        self.evaluation_llm = openai.AsyncOpenAI(api_key=cfg.env.openai_api_key)
        self.tasks: list[BenchmarkTask] = []
        self.results: list[BenchmarkResult] = []

        # Get LLM provider and model from the config object
        self.llm_provider = cfg.llm.provider
        self.llm_model = cfg.llm.model_name

        # Initialize pipeline components
        self.get_log_dir()
        print("Initializing pipeline components...")

        print(
            f"Pipeline components initialized successfully! Using pass@{self.pass_at_k}"
        )

    @abstractmethod
    def load_tasks(self) -> list[BenchmarkTask]:
        """Load benchmark tasks from data files"""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def prepare_task_description(self, task: BenchmarkTask) -> tuple[str, str | None]:
        """Prepare task description and file path for the agent"""
        raise NotImplementedError("Subclasses must implement this method")

    def get_log_dir(self) -> Path:
        """Get the log directory for the current benchmark and model."""
        return Path(self.cfg.output_dir)

    async def run_single_task(self, task: BenchmarkTask, cfg) -> BenchmarkResult:
        """
        Run inference for a single benchmark task with pass@k support

        Args:
            task: BenchmarkTask object

        Returns:
            BenchmarkResult object
        """
        print(f"Processing task {task.task_id} with pass@{self.pass_at_k}")

        result = BenchmarkResult(
            task_id=task.task_id,
            task_question=task.task_question,
            ground_truth=task.ground_truth,
            file_path=task.file_path,
            model_response="",
            model_boxed_answer="",
            status="pending",
            metadata=task.metadata.copy(),
            k_value=self.pass_at_k,
        )

        found_correct_answer = False

        # Print debug info about log directory
        print(f"  Current log directory: {self.output_dir}")

        try:
            # Prepare task
            task_description, task_file_path = self.prepare_task_description(task)

            # Get base_url and api_key from cfg if available
            base_url = self.cfg.get("base_url", None)
            api_key = self.cfg.get("api_key", None)

            # Run up to k attempts (with early stopping when correct answer found)
            for attempt in range(1, self.pass_at_k + 1):
                print(f"  Attempt {attempt}/{self.pass_at_k} for task {task.task_id}")

                attempt_result = self.scan_latest_attempt(task, attempt)
                # Run inference if no existing result
                if attempt_result["status"] in (
                    TaskStatus.PENDING,
                    TaskStatus.RUN_FAILED,
                ):
                    final_boxed_answer = ""
                    response = ""
                    max_retries = 6
                    retry_count = 0
                    while final_boxed_answer == "" and retry_count < max_retries:
                        try:
                            run_result = await run_backend(
                                task_file_path=task_file_path,
                                task_description=task_description,
                                log_path=self.output_dir
                                / f"{task.task_id}_attempt_{attempt}",
                                tags=[
                                    *self.cfg.tags,
                                    f"query_id_{task.task_id}",
                                    f"retry_{retry_count}",
                                ],
                                task_id=task.task_id,
                                gt=task.ground_truth,
                                base_url=base_url,
                                api_key=api_key,
                                model_name=cfg.llm.model_name,
                            )
                            response = _last_assistant_content(run_result.messages)
                            final_boxed_answer = run_result.final_answer or ""
                            log_file_path = run_result.log_path
                            retry_count += 1
                            if final_boxed_answer == "" and retry_count < max_retries:
                                print(
                                    f"    Empty boxed answer, retrying ({retry_count}/{max_retries})..."
                                )
                        except Exception as e:
                            attempt_result["status"] = TaskStatus.RUN_FAILED
                            attempt_result["error_message"] = str(e)
                            print(f"    Error in attempt {attempt}: {e}")
                            break

                    if attempt_result.get("error_message"):
                        pass  # Exception occurred, skip processing
                    else:
                        attempt_result["model_response"] = response if response else ""

                        # Save response data to log file
                        import time

                        timestamp = int(time.time())
                        log_file_path = (
                            self.output_dir
                            / f"task_{task.task_id}_attempt_{attempt}_{timestamp}.json"
                        )
                        log_data = {
                            "output": attempt_result["model_response"],
                            "final_boxed_answer": final_boxed_answer or "",
                            "task_id": task.task_id,
                            "ground_truth": task.ground_truth,
                            "timestamp": timestamp,
                        }
                        with open(log_file_path, "w", encoding="utf-8") as f:
                            json.dump(log_data, f, indent=2, ensure_ascii=False)

                        attempt_result["log_file_path"] = log_file_path
                        if final_boxed_answer:
                            attempt_result["model_boxed_answer"] = final_boxed_answer
                            attempt_result["status"] = TaskStatus.RUN_COMPLETED
                        else:
                            attempt_result["model_boxed_answer"] = final_boxed_answer
                            attempt_result["status"] = TaskStatus.RUN_FAILED

                # Perform LLM verification if we have an answer and haven't verified yet
                if attempt_result["status"] == TaskStatus.RUN_COMPLETED:
                    print(f"    Verifying answer for attempt {attempt}...")
                    try:
                        evaluation_result = await verify_answer_for_datasets(
                            openai_client=self.evaluation_llm,
                            benchmark_name=self.benchmark_name,
                            question=task.task_question,
                            target=task.ground_truth,
                            predicted_answer=attempt_result["model_boxed_answer"],
                        )
                        attempt_result["llm_as_judge_result"] = evaluation_result
                        attempt_result["is_correct"] = evaluation_result == "CORRECT"
                        # trace.score(value=1.0 if attempt_result["is_correct"] else 0.0, name="correctness")

                        # Update the log file with verification result
                        if "log_file_path" in attempt_result and isinstance(
                            attempt_result["log_file_path"], Path
                        ):
                            await self._update_log_file_with_evaluation(
                                attempt_result["log_file_path"],
                                evaluation_result,
                                task.ground_truth,
                                attempt_result["is_correct"],
                            )

                        if attempt_result["is_correct"]:
                            print(f"    ✅ Attempt {attempt}: CORRECT!")
                            found_correct_answer = True
                        else:
                            print(
                                f"    ❌ Attempt {attempt}: INCORRECT ({evaluation_result})"
                            )

                    except Exception as e:
                        print(f"    Error verifying attempt {attempt}: {e}")
                        attempt_result["llm_as_judge_result"] = "ERROR"
                        attempt_result["is_correct"] = False

                if attempt_result["is_correct"]:
                    print(f"    ✅ Attempt {attempt}: CORRECT (cached)")
                    found_correct_answer = True
                elif attempt_result["llm_as_judge_result"]:
                    print(
                        f"    ❌ Attempt {attempt}: INCORRECT (cached: {attempt_result['llm_as_judge_result']})"
                    )
                else:
                    print(f"    ⚠️  Attempt {attempt}: No valid answer to verify")

                result.attempts.append(attempt_result)

                # Update main result with the first successful attempt or best attempt so far
                if attempt == 1 or (
                    attempt_result["status"] == TaskStatus.RUN_COMPLETED
                    and not result.model_boxed_answer
                ):
                    result.model_response = attempt_result["model_response"]
                    result.model_boxed_answer = attempt_result["model_boxed_answer"]
                    result.log_file_path = attempt_result["log_file_path"]
                    result.status = attempt_result["status"]
                    if attempt_result["error_message"] is not None:
                        result.error_message = attempt_result["error_message"]

                # Early stopping: if we found a correct answer, we can stop
                if found_correct_answer:
                    print(
                        f"    🎯 Found correct answer! Stopping early after {attempt} attempts."
                    )
                    break

        except Exception as e:
            result.error_message = str(e)
            result.status = "failed"
            print(f"Error processing task {task.task_id}: {e}")

        finally:
            result.pass_at_k_success = found_correct_answer

            # Set main result LLM judge result based on pass@k outcome
            if found_correct_answer:
                result.llm_as_judge_result = "PASS_AT_K_SUCCESS"
            else:
                result.llm_as_judge_result = "PASS_AT_K_FAILED"

            print(f"Task {task.task_id} completed with {len(result.attempts)} attempts")
            print(
                f"    Pass@{self.pass_at_k} result: {'✅ SUCCESS' if found_correct_answer else '❌ FAILED'}"
            )

        return result

    def scan_latest_attempt(self, task: BenchmarkTask, attempt: int) -> AttemptStats:
        """check filesystem for latest attempt"""
        attempt_result: AttemptStats = {
            "attempt_number": attempt,
            "model_response": "",
            "model_boxed_answer": "",
            "status": TaskStatus.PENDING,
            "log_file_path": None,
            "llm_as_judge_result": None,
            "is_correct": False,
            "error_message": None,
        }
        trace_filename_pattern = f"task_{task.task_id}_attempt_{attempt}_*.json"
        matched_logs = self.output_dir.glob(trace_filename_pattern)
        sorted_logs = sorted(matched_logs, reverse=True)
        if len(sorted_logs) == 0:
            return attempt_result
        latest_log = sorted_logs[0]
        attempt_result["status"] = TaskStatus.RUN_FAILED
        attempt_result["log_file_path"] = latest_log
        print(f"    Found existing log for attempt {attempt}: {latest_log.name}")

        with open(latest_log) as f:
            log_data = json.loads(f.read())
            if log_data.get("final_boxed_answer"):
                attempt_result["status"] = TaskStatus.RUN_COMPLETED
                attempt_result["model_boxed_answer"] = log_data["final_boxed_answer"]
                attempt_result["model_response"] = log_data.get("output", "")
                # Check if we already have LLM judge result in log
                if log_data.get("llm_as_judge_result"):
                    attempt_result["status"] = TaskStatus.RESULT_JUDGED
                    attempt_result["llm_as_judge_result"] = log_data[
                        "llm_as_judge_result"
                    ]
                    attempt_result["is_correct"] = (
                        log_data["llm_as_judge_result"] == "CORRECT"
                    )
                print(
                    f"    Loaded existing result: {attempt_result['model_boxed_answer']}"
                )
        return attempt_result

    async def run_parallel_inference(
        self, tasks: list[BenchmarkTask], max_concurrent: int = 3, cfg=None
    ) -> list[BenchmarkResult]:
        """Run inference on multiple tasks in parallel"""
        print(
            f"Running inference on {len(tasks)} tasks with max_concurrent={max_concurrent}"
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task, cfg):
            async with semaphore:
                return await self.run_single_task(task, cfg)

        # Run tasks in parallel
        results = await asyncio.gather(
            *[run_with_semaphore(task, cfg) for task in tasks], return_exceptions=True
        )

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Exception in task {tasks[i].task_id}: {result}")
                error_result = BenchmarkResult(
                    task_id=tasks[i].task_id,
                    task_question=tasks[i].task_question,
                    ground_truth=tasks[i].ground_truth,
                    file_path=tasks[i].file_path,
                    model_response="",
                    model_boxed_answer="",
                    status="failed",
                    metadata=tasks[i].metadata.copy(),
                    error_message=str(result),
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        self.results = processed_results
        return processed_results

    def save_results(self, output_path: Path) -> Path:
        """Save evaluation results to JSONL file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in self.results:
                f.write(result.model_dump_json() + "\n")

        print(f"Results saved to {output_path}")
        return output_path

    async def evaluate_accuracy(self) -> float:
        """Evaluate pass@k accuracy (verification already done in run_single_task)"""
        if not self.results:
            print("No results to evaluate")
            return 0.0

        print(
            f"Calculating pass@{self.pass_at_k} accuracy for {len(self.results)} results..."
        )

        correct_count = 0
        total_count = 0

        for result in self.results:
            total_count += 1

            # Display task results
            print(f"\nTask {result.task_id}:")
            print(f"  Attempts: {len(result.attempts)}")
            print(
                f"  Pass@{self.pass_at_k}: {'✅ SUCCESS' if result.pass_at_k_success else '❌ FAILED'}"
            )

            # Show details of each attempt
            for attempt in result.attempts:
                attempt_num = attempt.get("attempt_number", "?")
                judge_result = attempt.get("llm_as_judge_result", "NOT_VERIFIED")
                is_correct = attempt.get("is_correct", False)
                status_icon = (
                    "✅"
                    if is_correct
                    else "❌"
                    if judge_result != "NOT_VERIFIED"
                    else "⚠️"
                )
                print(f"    Attempt {attempt_num}: {status_icon} {judge_result}")
                if attempt.get("model_boxed_answer"):
                    print(f"      Answer: {attempt['model_boxed_answer']}")

            print("  " + "=" * 50)
            print(f"  Reference: {result.ground_truth}")
            print("  " + "=" * 50)

            if result.pass_at_k_success:
                correct_count += 1

        pass_at_k_accuracy = correct_count / total_count if total_count > 0 else 0.0

        print(f"\nPass@{self.pass_at_k} Final Results:")
        print(f"Tasks passed: {correct_count}/{total_count}")
        print(f"Pass@{self.pass_at_k} Accuracy: {pass_at_k_accuracy:.2%}")

        return pass_at_k_accuracy

    async def _update_log_file_with_evaluation(
        self,
        log_file_path: Path,
        evaluation_result: str,
        ground_truth: str,
        is_correct: bool,
    ):
        """Helper method to update log file with evaluation result"""
        try:
            log_file = Path(log_file_path)
            # Read existing data
            async with aiofiles.open(log_file, encoding="utf-8") as f:
                content = await f.read()
                log_data = json.loads(content)

            # Update with evaluation result
            log_data["llm_as_judge_result"] = evaluation_result
            log_data["is_correct"] = is_correct
            log_data["ground_truth"] = ground_truth
            # Write to a temporary file and then atomically replace
            temp_log_file = log_file.with_suffix(f"{log_file.suffix}.tmp")
            async with aiofiles.open(temp_log_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(log_data, indent=2))

            os.replace(temp_log_file, log_file)
            print(f"    Updated log file {log_file.name} with evaluation result.")
        except Exception as e:
            print(f"    Error updating log file {log_file_path}: {e}")


class JSONLDatasetEvaluator(BenchmarkEvaluator):
    """benchmark evaluator for Gaia like dataset."""

    def __init__(
        self,
        data_dir: str,
        benchmark_name: str,
        cfg: DictConfig,
        metadata_file: str,
        parse_func: Callable[[str], BenchmarkTask],
        filter_func: Callable[[BenchmarkTask], bool],
    ):
        """
        dataset format:
        - a FOLDER (`data_dir`) with a METADATA file (`metadata_file`) and many other binary files.
        - METADATA file are newline separated json objects, parsed by `parse_func` into `BenchmarkTask` objects.
        - `filter_func` is used to filter tasks based on a condition.
        - binary files are referenced by `BenchmarkTask.file_path`.

        Args:
            data_dir: Path to benchmark data directory
            benchmark_name: Name of the benchmark
            cfg: The Hydra configuration object
            parse_func: Function to parse a line of data into a BenchmarkTask object
            filter_func: Function to filter tasks based on a condition
        """
        super().__init__(data_dir=data_dir, benchmark_name=benchmark_name, cfg=cfg)
        self.metadata_file = self.data_dir / metadata_file
        self.parse_func = parse_func
        self.filter_func = filter_func
        self.tasks: list[BenchmarkTask] = []
        self.results: list[BenchmarkResult] = []

    def load_tasks(self) -> list[BenchmarkTask]:
        """
        Load benchmark tasks from metadata.jsonl

        Returns:
            List of BenchmarkTask objects
        """
        print(f"Loading tasks from {self.metadata_file}")

        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        tasks = []
        with open(self.metadata_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    task = self.parse_func(line.strip())
                    if task.file_path is not None:
                        pass
                    if self.filter_func(task):
                        tasks.append(task)

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {i + 1}: {e}")
                    continue
        tasks = tasks[: self.cfg.benchmark.execution.max_tasks]
        self.tasks = tasks
        print(f"Loaded {len(tasks)} tasks")
        return tasks

    def prepare_task_description(self, task: BenchmarkTask) -> tuple[str, str | None]:
        if task.file_path is None:
            return task.task_question, []

        path = Path(task.file_path)
        # check if task.file_path is a relative path
        if path.is_absolute():
            return task.task_question, [str(path.resolve())]

        # 构建完整文件路径：数据目录 + 相对路径
        full_file_path = Path(self.data_dir) / path
        return task.task_question, [str(full_file_path.resolve())]


async def entrypoint(cfg, data_dir="") -> float:
    """
    Main entry point for running benchmarks with Hydra.
    """

    def parse_func(x: str) -> BenchmarkTask:
        data = json.loads(x)
        return BenchmarkTask(
            task_id=data["task_id"],
            task_question=data["Question"],
            ground_truth=data["Final answer"],
            file_path=None if data.get("file_name") == "" else data.get("file_name"),
            metadata=data.get("Annotator Metadata", {}),
            Level=data.get("Level", ""),
        )

    def filter_func(x: BenchmarkTask) -> bool:
        if len(cfg.benchmark.data.whitelist) > 0:
            return x.task_id in cfg.benchmark.data.whitelist
        else:
            return True

    evaluator = JSONLDatasetEvaluator(
        data_dir=cfg.benchmark.data.data_dir,
        benchmark_name=cfg.benchmark.name,
        cfg=cfg,
        metadata_file=cfg.benchmark.data.metadata_file,
        parse_func=parse_func,
        filter_func=filter_func,
    )

    """
    Run the full benchmark evaluation process
    """
    print(f"Starting evaluation for benchmark: {cfg.benchmark.name}")
    print(f"LLM Provider: {evaluator.llm_provider}")
    print(f"LLM Model: {evaluator.llm_model}")

    level = cfg.level
    # Load tasks
    tasks = evaluator.load_tasks()
    tasks = [t for t in tasks if t.Level == level]

    import glob

    folder_path = cfg.output_dir + "/*_attempt_1"
    runned = glob.glob(folder_path)

    runned_ = [i.split(cfg.output_dir + "/")[1].split("_attempt_1")[0] for i in runned]
    # tasks = [t for t in tasks if t.Level == level and t.task_id not in runned_]
    tasks = [t for t in tasks if t.Level == 1 and t.task_id not in runned_]

    if len(evaluator.tasks) == 0:
        print("No tasks loaded. Exiting.")
        return 0.0

    # Run inference
    print(
        f"\nStarting parallel inference with {cfg.benchmark.execution.max_concurrent} concurrent tasks..."
    )
    print(f"Using pass@{evaluator.pass_at_k} evaluation...")
    await evaluator.run_parallel_inference(
        tasks, max_concurrent=cfg.benchmark.execution.max_concurrent, cfg=cfg
    )

    # Evaluate accuracy
    print("Evaluating accuracy...")
    accuracy = await evaluator.evaluate_accuracy()
    print(f"\nOverall pass@{evaluator.pass_at_k} accuracy: {accuracy:.2%}")
    # Save results

    output_filename = "benchmark_results.jsonl"

    # Construct the full path in the correct log directory
    log_dir = evaluator.output_dir
    results_path = log_dir / output_filename

    evaluator.save_results(results_path)
    print(f"\nEvaluation completed! Results saved to {results_path}")
    # save accuracy to a file
    accuracy_file = (
        results_path.parent
        / f"{results_path.stem}_pass_at_{evaluator.pass_at_k}_accuracy.txt"
    )
    with open(accuracy_file, "w") as f:
        f.write(f"{accuracy:.2%}")

    return accuracy


def main():
    dotenv.load_dotenv()

    # Create configuration using OmegaConf
    cfg = OmegaConf.create(
        {
            "benchmark": {
                "name": "gaia-validation",
                "data": {
                    "data_dir": "customized_areal/dataset/gaia-benchmark/gaia/2023/validation",
                    "metadata_file": "metadata.jsonl",
                    "whitelist": [],
                },
                "execution": {"max_concurrent": 10, "max_tasks": 166, "pass_at_k": 1},
            },
            "llm": {
                "provider": "openai",
                # "model_name": "openrouter/gpt-5",
                # "model_name": "openai-compatible/gpt-5",
                # "model_name": "openrouter/qwen/qwen3.5-9b",
                "model_name": "openrouter/qwen/qwen3-vl-8b-thinking",
                # "model_name": "openrouter/qwen/qwen3-32b",
                "enable_thinking": False,
                "reasoning_effort": "low",
                "stream": False,
            },
            "env": {"openai_api_key": ""},
            "level": 1,
            "user_id": "62ec5137-d121-4c8c-b175-ee165bdf38e4",
            "agent_id": os.environ.get("main_agent_id", ""),
            "backend_mode": True,
            "base_url": "https://openrouter.ai/api/v1",  # Set your proxy base URL here or via CLI
            "api_key": "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",  # Set your API key here or via CLI
            # "base_url": "http://10.254.94.128:8443/service-large-544-1773728352034/llm/v1",  # Set your proxy base URL here or via CLI
            # "api_key": "Rl44TWGlj7Nn06txRhLrmgLf888A768jvxZc6Xm1gD7mtcrz2Vrg0pNH8rdP8mg688jl8Xdcq7MSB7Anzp8pf8XgnK7168R2267ZBS5dSlzbGhr6rwB5t6ZcP5wn6w7t",  # Set your API key here or via CLI
        }
    )

    # Compute derived values
    cfg.tags = [
        f"{cfg.benchmark.name}",
        f"{cfg.llm.model_name}",
        "base_0517",
        # "compression_1w",
        f"level_{cfg.level}",
    ]

    cfg.output_dir = f"logs/{'-'.join(cfg.tags[:-1])}/{cfg.tags[-1]}"

    asyncio.run(entrypoint(cfg))


if __name__ == "__main__":
    main()
