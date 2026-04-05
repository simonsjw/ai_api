-- =============================================
-- ai_api Test Database Permissions
-- Run as PostgreSQL superuser AFTER createTestDB.sql
-- =============================================

-- Assumes user name 'testuser' for testdb.
-- (this is typically defined in your infopypg configuration, .env, or pyproject.toml)
DO $$
DECLARE
    test_user TEXT := 'testuser'; 
BEGIN
    -- Schema-level privileges required for partition creation
    EXECUTE format('GRANT USAGE, CREATE ON SCHEMA public TO %I', test_user);

    -- Full DML on core tables (already granted previously)
    EXECUTE format('GRANT SELECT, INSERT, UPDATE ON TABLE providers TO %I', test_user);
    EXECUTE format('GRANT SELECT, INSERT, UPDATE ON TABLE requests TO %I', test_user);
    EXECUTE format('GRANT SELECT, INSERT, UPDATE ON TABLE responses TO %I', test_user);
    EXECUTE format('GRANT SELECT, INSERT ON TABLE logs TO %I', test_user);

    -- Sequences (for IDENTITY columns and partitions)
    EXECUTE format('GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO %I', test_user);

    RAISE NOTICE 'Full test-database privileges granted to user %', test_user;
END $$;

-- Create partition for TODAY (2026-04-05) – matches current date
CREATE TABLE IF NOT EXISTS requests_2026_04_05
    PARTITION OF requests
    FOR VALUES FROM ('2026-04-05 00:00:00+00') TO ('2026-04-06 00:00:00+00');

CREATE TABLE IF NOT EXISTS responses_2026_04_05
    PARTITION OF responses
    FOR VALUES FROM ('2026-04-05 00:00:00+00') TO ('2026-04-06 00:00:00+00');

CREATE TABLE IF NOT EXISTS logs_2026_04_05
    PARTITION OF logs
    FOR VALUES FROM ('2026-04-05 00:00:00+00') TO ('2026-04-06 00:00:00+00');

-- Seed providers (prevents first-run lookup failures)
INSERT INTO providers (name, description)
VALUES ('xai', 'xAI Grok API provider')
ON CONFLICT (name) DO NOTHING;

-- Verification
-- SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;
-- SELECT * FROM providers;
