sudo -u postgres psql --no-psqlrc -h localhost -d postgres  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'testdb' AND pid <> pg_backend_pid();"
sudo -u postgres psql --no-psqlrc -h localhost -d postgres -c "DROP DATABASE IF EXISTS testdb;"
sudo -u postgres psql --no-psqlrc -h localhost -d postgres -c "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'testuser') THEN CREATE ROLE testuser WITH LOGIN PASSWORD 'testpass'; END IF; END \$\$;"
sudo -u postgres psql --no-psqlrc -h localhost -d postgres c "CREATE DATABASE testdb OWNER testuser;"

# Then run sql.
PGPASSWORD=testpass psql --no-psqlrc -h localhost -d postgres -U testuser -d testdb -f createTestDB_schema.sql
