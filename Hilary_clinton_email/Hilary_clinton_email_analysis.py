
#"But her emails..."
#
# In this problem, you'll show your SQL and Pandas chops on the dataset consisting of Hilary Rodham Clinton's emails!


import sys
print("=== Python version info ===\n{}".format(sys.version))

import sqlite3 as db
print("\n=== sqlite3 version info: {} ===".format(db.version))

from IPython.display import display
import pandas as pd
import numpy as np

def peek_table (db, name, num=5):
    """
    Given a database connection (`db`), prints both the number of
    records in the table as well as its first few entries.
    """
    count = '''select count (*) FROM {table}'''.format (table=name)
    peek = '''select * from {table} limit {limit}'''.format (table=name, limit=num)

    print ("Total number of records:", pd.read_sql_query (count, db)['count (*)'].iloc[0], "\n")

    print ("First {} entries:".format (num))
    display (pd.read_sql_query (peek, db))

def list_tables (conn):
    """Return the names of all visible tables, given a database connection."""
    query = """select name from sqlite_master where type = 'table';"""
    c = conn.cursor ()
    c.execute (query)
    table_names = [t[0] for t in c.fetchall ()]
    return table_names

def tbc (X):
    var_names = sorted (X.columns)
    Y = X[var_names].copy ()
    Y.sort_values (by=var_names, inplace=True)
    Y.set_index ([list (range (0, len (Y)))], inplace=True)
    return Y

def tbeq(A, B):
    A_c = tbc(A)
    B_c = tbc(B)
    return A_c.eq(B_c).all().all()

DATA_PATH = "./resource/asnlib/publicdata/"
conn = db.connect ('{}hrc.db'.format(DATA_PATH))

print ("List of tables in the database:", list_tables (conn))
peek_table (conn, 'Emails')
peek_table (conn, 'EmailReceivers', num=3)
peek_table (conn, 'Persons')

sql_string = 'SELECT Id, Name FROM Persons'

Persons = pd.read_sql_query(sql_string, conn)

assert 'Persons' in globals ()
assert type (Persons) is type (pd.DataFrame ())
assert len (Persons) == 513

print ("Five random people from the `Persons` table:")
display (Persons.iloc[np.random.choice (len (Persons), 5)])

print ("\n(Passed!)")


Sender = pd.read_sql_query('SELECT Id, SenderPersonId From Emails', conn)
Receiver = pd.read_sql_query('SELECT EmailId, PersonId From EmailReceivers', conn)
Sender = Sender[Sender['SenderPersonId'] != '']

solution_df = pd.merge(Sender, Receiver, left_on='Id', right_on='EmailId').groupby(['SenderPersonId', 'PersonId']).size().reset_index()
solution_df.columns = ['Sender','Receiver','Frequency']

CommEdges = solution_df.sort_values('Frequency', ascending=False)


# Read what we believe is the exact result (up to permutations)
CommEdges_soln = pd.read_csv ('{}CommEdges_soln.csv'.format(DATA_PATH))

# Check that we got a data frame of the expected shape:
assert 'CommEdges' in globals ()
assert type (CommEdges) is type (pd.DataFrame ())
assert len (CommEdges) == len (CommEdges_soln)
assert set (CommEdges.columns) == set (['Sender', 'Receiver', 'Frequency'])

# Check that the results are sorted:
non_increasing = (CommEdges['Frequency'].iloc[:-1].values >= CommEdges['Frequency'].iloc[1:].values)
assert non_increasing.all ()

print ("Top 5 communicating pairs:")
display (CommEdges.head ())

assert tbeq (CommEdges, CommEdges_soln)
print ("\n(Passed!)")

G = CommEdges.rename(columns={'Sender':'A','Receiver':'B'})
GT = CommEdges.rename(columns={'Sender':'B','Receiver':'A'})

H = pd.merge(G,GT,on=['A','B'], suffixes=['_G','_GT'])
H['Frequency'] = H['Frequency_G']+H['Frequency_GT']
del H['Frequency_G']
del H['Frequency_GT']

CommPairs = H
CommPairs_soln = pd.read_csv ('{}CommPairs_soln.csv'.format(DATA_PATH))

assert 'CommPairs' in globals ()
assert type (CommPairs) is type (pd.DataFrame ())
assert len (CommPairs) == len (CommPairs_soln)

print ("Most frequently communicating pairs:")
display (CommPairs.sort_values (by='Frequency', ascending=False).head (10))

assert tbeq (CommPairs, CommPairs_soln)
print ("\n(Passed!)")

CommPairsNamed = CommPairs.copy ()
CommPairsNamed = pd.merge(CommPairsNamed,Persons,left_on='A',right_on='Id')
CommPairsNamed.rename(columns={'Name':'A_name'}, inplace=True)
del CommPairsNamed['Id']
CommPairsNamed = pd.merge(CommPairsNamed,Persons,left_on='B',right_on='Id')
CommPairsNamed.rename(columns={'Name':'B_name'}, inplace=True)
del CommPairsNamed['Id']
CommPairsNamed

CommPairsNamed_soln = pd.read_csv ('{}CommPairsNamed_soln.csv'.format(DATA_PATH))

assert 'CommPairsNamed' in globals ()
assert type (CommPairsNamed) is type (pd.DataFrame ())
assert set (CommPairsNamed.columns) == set (['A', 'A_name', 'B', 'B_name', 'Frequency'])

print ("Top few entries:")
CommPairsNamed.sort_values (by=['Frequency', 'A', 'B'], ascending=False, inplace=True)
display (CommPairsNamed.head (10))

assert tbeq (CommPairsNamed, CommPairsNamed_soln)
print ("\n(Passed!)")



conn.close ()
