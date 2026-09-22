import sqlite3
import unittest

from sql_generator import DemoClient, compile_query, generate_query, read_schema
from typellm import compile_json_schema


class SQLTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.executescript("CREATE TABLE users(id INTEGER, name TEXT, age INTEGER, balance REAL, payload BLOB); INSERT INTO users VALUES(1, 'Alice', 25, 12.5, NULL), (2, 'Bob', 17, 0.0, NULL);")
        self.schema = read_schema(self.db)

    def tearDown(self):
        self.db.close()

    def test_staged_schema_and_execution(self):
        result = generate_query(DemoClient(), self.schema, 'Names of users older than 18')
        for stage in result['trace']:
            compile_json_schema({'type': 'object', 'properties': stage['questions']})
        self.assertEqual(result['trace'][-1]['questions']['value']['type'], 'integer')
        self.assertEqual(result['trace'][1]['questions']['select_column']['enum'], ['id', 'name', 'age', 'balance'])
        self.assertEqual(self.db.execute(result['sql'], result['params']).fetchall(), [('Alice',)])

    def test_text_is_bound_not_interpolated(self):
        attack = "Alice' OR 1=1 --"
        sql, params = compile_query(self.schema, {'table':'users', 'select_column':'name',
            'filter':{'column':'name','operator':'eq','value':attack}})
        self.assertNotIn(attack, sql)
        self.assertEqual(params, [attack, 100])
        self.assertEqual(self.db.execute(sql, params).fetchall(), [])

    def test_quoted_identifiers(self):
        self.db.execute('CREATE TABLE "a""b" ("x""y" TEXT)')
        sql, params = compile_query(read_schema(self.db), {'table':'a"b','select_column':'x"y'})
        self.assertEqual(self.db.execute(sql, params).fetchall(), [])

    def test_reject_invalid_values_names_operators_limits(self):
        for value in [True, 18.5, '18', None, 2**63]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                compile_query(self.schema, {'table':'users','select_column':'name', 'filter':{'column':'age','operator':'gt','value':value}})
        for kind, value in [('balance', float('nan')), ('balance', float('inf')), ('name', 12)]:
            with self.assertRaises(ValueError):
                compile_query(self.schema, {'table':'users','select_column':'name', 'filter':{'column':kind,'operator':'eq','value':value}})
        for plan in [
            {'table':'users; DROP TABLE users','select_column':'name'},
            {'table':'users','select_column':'missing'},
            {'table':'users','select_column':'name','filter':{'column':'name','operator':'gt','value':'A'}},
            {'table':'users','select_column':'name','filter':{'column':'age','operator':'OR 1=1','value':18}},
        ]:
            with self.assertRaises(ValueError): compile_query(self.schema, plan)
        for limit in [0, -1, 1001, True, '10']:
            with self.assertRaises(ValueError): compile_query(self.schema, {'table':'users','select_column':'name'}, limit)

    def test_no_filter_and_null_need_no_literal(self):
        class Fake:
            def __init__(self, answers): self.answers = iter(answers)
            def generate(self, **kwargs): return next(self.answers)
        result = generate_query(Fake([{'table':'users'}, {'select_column':'name','has_filter':False}]), self.schema, 'Names')
        self.assertEqual(len(result['trace']), 2)
        self.assertNotIn('WHERE', result['sql'])
        result = generate_query(Fake([{'table':'users'}, {'select_column':'name','has_filter':True}, {'column':'age'}, {'operator':'is_null'}]), self.schema, 'Users with missing age')
        self.assertEqual(len(result['trace']), 4)
        self.assertIn('IS NULL', result['sql'])
        self.assertEqual(result['params'], [100])

    def test_number_and_text_generate_correct_value_schema(self):
        for column, kind, value in [('balance','number',12.5), ('name','string','Alice')]:
            class Fake:
                def __init__(self):
                    self.answers=iter([{'table':'users'},{'select_column':'name','has_filter':True},{'column':column},{'operator':'eq'},{'value':value}])
                def generate(self, **kwargs): return next(self.answers)
            result=generate_query(Fake(),self.schema,'Find Alice')
            self.assertEqual(result['trace'][-1]['questions']['value']['type'],kind)
            self.assertEqual(self.db.execute(result['sql'], result['params']).fetchall(), [('Alice',)])


if __name__ == '__main__':
    unittest.main()
