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
    -- Schema usage (required for all objects)
    EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', test_user);

    -- providers table (critical for _get_or_create_provider_id)
    EXECUTE format('GRANT SELECT, INSERT, UPDATE ON TABLE providers TO %I', test_user);
    EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE providers_id_seq TO %I', test_user);

    -- Main persistence tables (full test coverage)
    EXECUTE format('GRANT SELECT, INSERT, UPDATE ON TABLE requests TO %I', test_user);
    EXECUTE format('GRANT SELECT, INSERT, UPDATE ON TABLE responses TO %I', test_user);
    EXECUTE format('GRANT SELECT, INSERT ON TABLE logs TO %I', test_user);

    -- All sequences (for IDENTITY columns on partitioned tables)
    EXECUTE format('GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO %I', test_user);

    RAISE NOTICE 'Permissions successfully granted to user % for ai_api test database', test_user;
END $$;

-- Optional: Seed the providers table (prevents first-run INSERT)
INSERT INTO providers (name, description)
VALUES ('xai', 'xAI Grok API provider')
ON CONFLICT (name) DO NOTHING;

-- Verification (run these to confirm)
-- \dt
-- SELECT * FROM providers;
-- SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;
