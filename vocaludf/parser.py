# fourFn.py
#
# Demonstration of the pyparsing module, implementing a simple 4-function expression parser,
# with support for scientific notation, and symbols for e and pi.
# Extended to add exponentiation and simple built-in functions.
# Extended test cases, simplified pushFirst method.
# Removed unnecessary expr.suppress() call (thanks Nathaniel Peterson!), and added Group
# Changed fnumber to use a Regex, which is now the preferred method
# Reformatted to latest pypyparsing features, support multiple and variable args to functions
#
# Copyright 2003-2019 by Paul McGuire
#
from pyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    Regex,
    ParseException,
    ParseSyntaxException,
    CaselessKeyword,
    Suppress,
    delimitedList,
    Opt,
    Dict,
    QuotedString,
    Empty,
)
import math
import pyparsing as pp
from pyparsing import pyparsing_common as ppc
import pprint
import json

def parse():
    """
    "Duration((Eastward2(o1), Eastward4(o0)), 5); Duration((Eastward2(o1), Eastward3(o0)), 5)"
    "(Color(o0, 'red'), Far(o0, o1, 3), Shape(o1, 'cylinder')); (Near(o0, o1, 1), RightQuadrant(o2), TopQuadrant(o2))"

    seqop   :: ';'
    conjop  :: ','
    var     :: 'o' '0'..'9'+
    value  :: string | number | TRUE | FALSE
    posint  :: '1'..'9' '0'..'9'*
    predicate :: fn '(' var [ ',' var ] [ ',' value ] ')'
    unpar_preds :: predicate [ conjop predicate ]*
    par_preds  :: '(' unpar_preds ')'
    duration :: 'Duration(' par_preds | predicate ',' posint ')'
    graph   :: duration | par_preds | predicate
    expr    :: duration | unpar_preds | graph [ seqop graph ]+
    stmt   :: expr | '(' expr ')'
    """

    seq =  Literal(";")
    lpar, rpar, comma = map(Suppress, "(),")
    TRUE = CaselessKeyword("true")
    FALSE = CaselessKeyword("false")
    var = Word("o", alphanums)
    posint = Regex(r"[1-9]\d*").set_parse_action(lambda t: int(t[0]))
    string = QuotedString("'") | QuotedString('"')
    number = ppc.number()
    value = string | number | TRUE | FALSE
    fn_name = Word(alphas, alphanums + "_")
    fn_args = Group(var - Opt(comma + var))
    predicate = Group(fn_name("predicate") - lpar - fn_args('variables') + Opt(comma + value)('parameter').set_parse_action(lambda t: t[0] if t else None) - rpar)
    bracketed_graph = (lpar - delimitedList(predicate) - rpar)
    unbracketed_graph = delimitedList(predicate)
    # unbracketed_graph = predicate - Opt(comma + predicate)
    unbracketed_one_graph = predicate
    graph = bracketed_graph | unbracketed_graph
    duration = Group("Duration" - lpar - (unbracketed_one_graph | bracketed_graph)("scene_graph") - "," - posint("duration_constraint") - rpar) | Group(graph("scene_graph") + Empty()("duration_constraint").set_parse_action(lambda t: 1))
    expr = Group(delimitedList(duration, delim=seq))
    bnf = expr("query")
    return bnf

if __name__ == "__main__":
    def test(s, expected=None):
        try:
            result = parse().parseString(s, parseAll=True).as_dict()
            result_as_list = parse().parseString(s, parseAll=True).asList()
            # results_dump = parse().parseString(s, parseAll=True).dump()
        except (ParseException, ParseSyntaxException) as err:
            print(err.explain())
        except Exception as e:
            print(s, "failed:\n", str(e))
        else:
            if expected is None:
                print(s, "->", result)
                print(result_as_list)
            elif result != expected:
                print(f"failed:\n[expected] {expected}\n[result] {result}")
                # print(results_as_list)
                # print(results_dump)
            else:
                print("success")
                # print(results_as_list)
                # print(results_dump)
    # [{'scene_graph': [{'predicate': 'Color', 'parameter': 'red', 'variables': ['o0']}, {'predicate': 'Far', 'parameter': 3.0, 'variables': ['o0', 'o1']}, {'predicate': 'Shape', 'parameter': 'cylinder', 'variables': ['o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Near', 'parameter': 1.0, 'variables': ['o0', 'o1']}, {'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'TopQuadrant', 'parameter': None, 'variables': ['o2']}], 'duration_constraint': 1}]
    test_data = [
        "(Color(o0, 'red'), Far(o0, o1, 3), Shape(o1, 'cylinder')); (Near(o0, o1, 1), RightQuadrant(o2), TopQuadrant(o2))",
    ]
    test("Color(o0, 1)", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 1}], 'duration_constraint': 1}]})
    test("Color(o0, 'red')", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}], 'duration_constraint': 1}]})
    test('Color(o0, "red")', {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}], 'duration_constraint': 1}]})
    test("Far(o0, o1, 3)", {'query': [{'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3}], 'duration_constraint': 1}]})
    test("RightQuadrant(o2)", {'query': [{'scene_graph': [{'predicate': 'RightQuadrant', 'variables': ['o2']}], 'duration_constraint': 1}]})
    test("(Color(o0, 'red'), Far(o0, o1, 3))", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}, {'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3}], 'duration_constraint': 1}]})
    test("(Color(o0, 'red'), Far(o0, o1, 3), Shape(o1, 'cylinder'))", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}, {'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3}, {'predicate': 'Shape', 'variables': ['o1'], 'parameter': 'cylinder'}], 'duration_constraint': 1}]})
    test("Color(o0, 'red'), Far(o0, o1, 3), Shape(o1, 'cylinder')", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}, {'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3}, {'predicate': 'Shape', 'variables': ['o1'], 'parameter': 'cylinder'}], 'duration_constraint': 1}]})
    test("Color(o0, 'red'); Far(o0, o1, 3)", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3}], 'duration_constraint': 1}]})
    test("(Color(o0, 'red'), Far(o0, o1, 3)); Shape(o1, 'cylinder')", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}, {'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Shape', 'variables': ['o1'], 'parameter': 'cylinder'}], 'duration_constraint': 1}]})
    test("Color(o0, 'red'), Far(o0, o1, 3); Shape(o1, 'cylinder')", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}, {'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Shape', 'variables': ['o1'], 'parameter': 'cylinder'}], 'duration_constraint': 1}]})
    test("(Color(o0, 'red'), Far(o0, o1, 3)); Duration(Shape(o1, 'cylinder'), 3)", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}, {'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Shape', 'variables': ['o1'], 'parameter': 'cylinder'}], 'duration_constraint': 3}]})
    test("Duration((Color(o0, 'red'), Far(o0, o1, 3)), 5); Duration(Shape(o1, 'cylinder'), 3)", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}, {'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3}], 'duration_constraint': 5}, {'scene_graph': [{'predicate': 'Shape', 'variables': ['o1'], 'parameter': 'cylinder'}], 'duration_constraint': 3}]})
    test("(Color(o0, 'red'), Far(o0, o1, 3), Shape(o1, 'cylinder')); (Near(o0, o1, 1), RightQuadrant(o2), TopQuadrant(o2))", {'query': [{'scene_graph': [{'predicate': 'Color', 'variables': ['o0'], 'parameter': 'red'}, {'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3}, {'predicate': 'Shape', 'variables': ['o1'], 'parameter': 'cylinder'}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1}, {'predicate': 'RightQuadrant', 'variables': ['o2']}, {'predicate': 'TopQuadrant', 'variables': ['o2']}], 'duration_constraint': 1}]})


    # Syntax error
    test("Far(o0, o1, o2, 3)")
    test("Duration(Color(o0, 'red'), Far(o0, o1, 3), 5)")
    test("(Far(o0, o1); Near(o0, o1); Far(o0, o1))")