-- Connect to your test database first:
-- psql -h 127.0.0.1 -p 5432 -U postgres -d testdb
--------------------
-- do this if you keep getting a fail on the last test.
-- for some reason the automatic log table creation doesn't work.
-- this is a work around.
-------------------
-- 1. Create the parent table (partitioned by tstamp)
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

-- 2. Create the default index used by queries
CREATE INDEX IF NOT EXISTS ix_logs_tstamp ON logs (tstamp);

-- 3. Add the trigger that guarantees tstamp is never NULL
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

CREATE OR REPLACE TRIGGER logs_set_tstamp
  BEFORE INSERT ON logs
  FOR EACH ROW
  EXECUTE FUNCTION set_logs_tstamp ();

