"""Migration framework for YDB.

Components:
- base: the abstract migration API every migration file implements
- discovery: finds migration files on disk and loads their classes
- planning: compares discovered files with the recorded history
- repository: reads and writes the ``schema_migrations`` history table
- schema: read-only inspection of the live schema through the Table Service
- operations: idempotent ``ALTER TABLE`` helpers
- verification: checks declared artifacts against the live schema
- executor: runs one migration and records its status
- manager: orchestrates discovery, planning, execution and the final check
- logging: console output of the framework
- cli: command-line entry point (``status``, ``up``)
"""
