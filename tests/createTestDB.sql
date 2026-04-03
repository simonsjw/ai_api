-- =============================================
-- Complete Test Database Setup for ai_api
-- Run as the PostgreSQL superuser (postgres):
--     psql -h 127.0.0.1 -p 5432 -U postgres -f setup_test_db.sql
-- =============================================
-- Ensure we start from the default postgres database
\c postgres
-- 1. Create the test user (idempotent)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT
    FROM
      pg_catalog.pg_roles
    WHERE
      rolname = 'testuser') THEN
  CREATE USER testuser WITH PASSWORD 'testpass';
END IF;
END
$$;

-- 2. Create the test database owned by testuser (idempotent)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT
    FROM
      pg_database
    WHERE
      datname = 'test') THEN
  CREATE DATABASE test OWNER testuser;
END IF;
END
$$;

-- 3. Switch to the test database
\c test
-- 4. Grant required privileges to testuser
GRANT CONNECT ON DATABASE test TO testuser;

GRANT CREATE, USAGE ON SCHEMA public TO testuser;

-- 5. Create the partitioned logs table (your original definition)
CREATE TABLE IF NOT EXISTS logs (
  idx bigint GENERATED ALWAYS AS IDENTITY,
  tstamp timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
  loglvl text NOT NULL,
  logger text NOT NULL,
  message text NOT NULL,
  obj jsonb,
  PRIMARY KEY (idx, tstamp)
)
PARTITION BY RANGE (tstamp);

-- 6. Create the index used by queries
CREATE INDEX IF NOT EXISTS ix_logs_tstamp ON logs (tstamp);

-- 7. Create the trigger function that guarantees tstamp is never NULL
CREATE OR REPLACE FUNCTION set_logs_tstamp ()
  RETURNS TRIGGER
  AS $$
BEGIN
  IF NEW.tstamp IS NULL THEN
    NEW.tstamp := CURRENT_TIMESTAMP;
  END IF;
  RETURN NEW;
END;
$$
LANGUAGE plpgsql;

-- 8. Create the trigger
CREATE OR REPLACE TRIGGER logs_set_tstamp
  BEFORE INSERT ON logs
  FOR EACH ROW
  EXECUTE FUNCTION set_logs_tstamp ();

-- 9. Grant table-level permissions to testuser
GRANT SELECT, INSERT ON TABLE logs TO testuser;

-- Optional: add a descriptive comment
COMMENT ON TABLE logs IS 'AI API logging table - managed by infopypg logger integration';

-- =============================================
-- End of script
-- =============================================
