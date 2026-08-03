import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    print(list(checkpointer.list({"configurable": {}})))
