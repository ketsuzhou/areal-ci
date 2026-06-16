  
  create table if not exists public.query_bank (
    query_id text primary key,
    query text not null,
    gold_answer text not null,
    evaluation_rubric text[] not null default '{}',
    used4train text[] not null default '{}',
    synthetic boolean default null,
    level text,
    task_id text,
    label text[] not null default '{}',
    file_paths text[] not null default '{}',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
  );

