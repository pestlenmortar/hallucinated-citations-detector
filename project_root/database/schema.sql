-- SQL schema defining tables, indexes, and constraints for the search database

CREATE TABLE IF NOT EXISTS papers (
    paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    normalized_title TEXT,
    authors TEXT,
    year INTEGER,
    venue TEXT,
    doi TEXT UNIQUE,
    abstract TEXT,
    embedding_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_normalized_title ON papers(normalized_title);
CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_year ON papers(year);
