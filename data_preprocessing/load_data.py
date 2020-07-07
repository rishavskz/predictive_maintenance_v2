import pandas.io.sql as sqlio


def load_data(connection, table):
    sql = "select * from {};".format(table)
    data = sqlio.read_sql_query(sql, connection)
    return data

