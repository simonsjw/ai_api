-- =============================================================================
-- createTestDB_schema.sql – Corrected schema for ai_api tests
-- PRIMARY KEY now includes tstamp (required for partitioned tables)
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

GRANT ALL PRIVILEGES ON DATABASE testdb TO testuser;
GRANT ALL ON SCHEMA public TO testuser;

-- Providers table
CREATE TABLE IF NOT EXISTS providers (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Requests table – partitioned daily (production naming)
CREATE TABLE IF NOT EXISTS requests (
    provider_id  INTEGER NOT NULL REFERENCES providers(id),
    endpoint     JSONB NOT NULL,
    request_id   UUID NOT NULL,
    request      JSONB NOT NULL,
    meta         JSONB NOT NULL,
    tstamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (request_id, tstamp)          -- must include partition key
) PARTITION BY RANGE (tstamp);

-- Responses table – partitioned daily (production naming)
CREATE TABLE IF NOT EXISTS responses (
    provider_id    INTEGER NOT NULL REFERENCES providers(id),
    endpoint       JSONB NOT NULL,
    request_id     UUID NOT NULL,
    request_tstamp TIMESTAMPTZ NOT NULL,
    response_id    UUID NOT NULL,
    response       JSONB NOT NULL,
    meta           JSONB NOT NULL,
    tstamp         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (response_id, tstamp),        -- must include partition key
    CONSTRAINT fk_responses_requests 
        FOREIGN KEY (request_id, request_tstamp) 
        REFERENCES requests (request_id, tstamp)
) PARTITION BY RANGE (tstamp);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO testuser;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO testuser;

DO $$
BEGIN
    RAISE NOTICE 'Test database "testdb" has been fully rebuilt with production partition naming.';
    RAISE NOTICE 'Connection: postgresql://testuser:testpass@localhost:5432/testdb';
    RAISE NOTICE 'You may now re-run: pytest test_grok_client.py';
END
$$;
