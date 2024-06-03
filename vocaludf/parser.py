from pyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    nums,
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
    ungroup,
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
    preds  :: par_preds | unpar_preds
    duration :: 'Duration(' par_preds | predicate ',' posint ')'
    graph   :: duration | preds
    expr    :: graph [ seqop graph ]*
    stmt   :: expr | '(' expr ')'
    """

    seq =  Literal(";")
    lpar, rpar, comma = map(Suppress, "(),")
    TRUE = CaselessKeyword("true")
    FALSE = CaselessKeyword("false")
    var = Word("o", nums)
    posint = Regex(r"[1-9]\d*").set_parse_action(lambda t: int(t[0]))
    string = QuotedString("'") | QuotedString('"')
    number = ppc.number()
    value = string | number | TRUE | FALSE
    fn_name = ungroup(~Literal("Duration") + Word(alphas, alphanums + "_"))
    fn_args = Group(var - Opt(comma + var))
    predicate = Group(fn_name("predicate") - lpar - fn_args('variables') - Opt(comma + value)('parameter').set_parse_action(lambda t: t[0] if t else None) - rpar)
    par_preds = lpar + delimitedList(predicate) + rpar
    unpar_preds = delimitedList(predicate)
    preds = par_preds | unpar_preds
    duration = Group("Duration" - lpar - (predicate | par_preds)("scene_graph") - "," - posint("duration_constraint") - rpar)
    graph = duration | Group((preds)("scene_graph") + Empty()("duration_constraint").set_parse_action(lambda t: 1))
    expr = Group(delimitedList(graph, delim=seq))
    bnf = expr("query") | (lpar + expr("query") + rpar)
    return bnf

def parse_udf():
    """
    var     :: 'o' '0'..'9'+
    value  :: string | number | TRUE | FALSE
    predicate :: fn '(' var [ ',' var ] [ ',' value ] ')'
    """
    lpar, rpar, comma = map(Suppress, "(),")
    TRUE = CaselessKeyword("true")
    FALSE = CaselessKeyword("false")
    var = Word("o", nums)
    string = QuotedString("'") | QuotedString('"')
    number = ppc.number()
    value = string | number | TRUE | FALSE
    fn_name = ungroup(~Literal("Duration") + ~Literal("object") + Word(alphas, alphanums + "_"))
    fn_args = Group(var - Opt(comma + var))
    udf = fn_name("fn_name") - lpar - fn_args('variables') - Opt(comma + value)('parameter').set_parse_action(lambda t: t[0] if t else None) - rpar | Literal("object")("fn_name") - lpar - Group(var)('variables') - comma - Word(alphas, alphanums + "_")('parameter').set_parse_action(lambda t: t[0] if t else None) - rpar
    return udf

if __name__ == "__main__":
    def test_udf(s, expected=None):
        try:
            result = parse_udf().parseString(s, parseAll=True).as_dict()
            result_as_list = parse_udf().parseString(s, parseAll=True).asList()
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
            else:
                print("success")

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
            else:
                print("success")
    # [{'scene_graph': [{'predicate': 'Color', 'parameter': 'red', 'variables': ['o0']}, {'predicate': 'Far', 'parameter': 3.0, 'variables': ['o0', 'o1']}, {'predicate': 'Shape', 'parameter': 'cylinder', 'variables': ['o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Near', 'parameter': 1.0, 'variables': ['o0', 'o1']}, {'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'TopQuadrant', 'parameter': None, 'variables': ['o2']}], 'duration_constraint': 1}]


    # unit tests for UDF
    test_udf("object(o1, oname)", {'fn_name': 'object', 'variables': ['o1'], 'parameter': 'oname'})
    test_udf("Color_red(o1)", {'fn_name': 'Color_red', 'variables': ['o1']})
    test_udf("Color_red(o1, o2)", {'fn_name': 'Color_red', 'variables': ['o1', 'o2']})
    test_udf("Color_red(o1, -1)", {'fn_name': 'Color_red', 'variables': ['o1'], 'parameter': -1})
    test_udf("Color_red(o1, o2, 3)", {'fn_name': 'Color_red', 'variables': ['o1', 'o2'], 'parameter': 3})
    test_udf("left(o, o1)", {'fn_name': 'left', 'variables': ['o', 'o1']})
    # Syntax error
    test_udf("left(a1)")
    test_udf("left(o, oa)")
    test_udf("Color_red(o1, o2, o3)")
    test_udf("Duration(o1)")

    # unit tests for DSL
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
    test("(Far(o0, o1); Near(o0, o1); Far(o0, o1))", {'query': [{'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Near', 'variables': ['o0', 'o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1']}], 'duration_constraint': 1}]})
    test("Duration(Near(o0, o1, 3), 5)", {'query': [{'scene_graph': [{'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 3}], 'duration_constraint': 5}]})
    test("(Duration(Near(o0, o1, 3), 5))", {'query': [{'scene_graph': [{'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 3}], 'duration_constraint': 5}]})
    # scene graph queries
    test("(Behind(o0, o1), BottomQuadrant(o0), Color(o0, 'brown'), Near(o0, o2, 1.0))", {'query': [{'scene_graph': [{'predicate': 'Behind', 'variables': ['o0', 'o1']}, {'predicate': 'BottomQuadrant', 'variables': ['o0']}, {'predicate': 'Color', 'variables': ['o0'], 'parameter': 'brown'}, {'predicate': 'Near', 'variables': ['o0', 'o2'], 'parameter': 1.0}], 'duration_constraint': 1}]})
    test("(BottomQuadrant(o0), Color(o0, 'yellow'), Near(o0, o1, 1.0), RightQuadrant(o0))", {'query': [{'scene_graph': [{'predicate': 'BottomQuadrant', 'variables': ['o0']}, {'predicate': 'Color', 'variables': ['o0'], 'parameter': 'yellow'}, {'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}, {'predicate': 'RightQuadrant', 'variables': ['o0']}], 'duration_constraint': 1}]})
    test("(Behind(o0, o1), Color(o1, 'brown'), LeftQuadrant(o2), Shape(o1, 'cylinder')); Duration(RightOf(o0, o1), 15); Duration((BottomQuadrant(o2), RightOf(o0, o2)), 10)", {'query': [{'scene_graph': [{'predicate': 'Behind', 'variables': ['o0', 'o1']}, {'predicate': 'Color', 'variables': ['o1'], 'parameter': 'brown'}, {'predicate': 'LeftQuadrant', 'variables': ['o2']}, {'predicate': 'Shape', 'variables': ['o1'], 'parameter': 'cylinder'}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'RightOf', 'variables': ['o0', 'o1']}], 'duration_constraint': 15}, {'scene_graph': [{'predicate': 'BottomQuadrant', 'variables': ['o2']}, {'predicate': 'RightOf', 'variables': ['o0', 'o2']}], 'duration_constraint': 10}]})
    test("Duration((RightOf(o0, o1), Shape(o1, 'cube')), 15); Duration((Near(o0, o2, 1.0), RightQuadrant(o0)), 15); Duration((Behind(o0, o2), Far(o1, o2, 3.0), Material(o1, 'rubber')), 10)", {'query': [{'scene_graph': [{'predicate': 'RightOf', 'variables': ['o0', 'o1']}, {'predicate': 'Shape', 'variables': ['o1'], 'parameter': 'cube'}], 'duration_constraint': 15}, {'scene_graph': [{'predicate': 'Near', 'variables': ['o0', 'o2'], 'parameter': 1.0}, {'predicate': 'RightQuadrant', 'variables': ['o0']}], 'duration_constraint': 15}, {'scene_graph': [{'predicate': 'Behind', 'variables': ['o0', 'o2']}, {'predicate': 'Far', 'variables': ['o1', 'o2'], 'parameter': 3.0}, {'predicate': 'Material', 'variables': ['o1'], 'parameter': 'rubber'}], 'duration_constraint': 10}]})
    test("(BottomQuadrant(o0), LeftOf(o0, o1)); LeftOf(o1, o2); (Behind(o1, o2), Color(o0, 'gray'), LeftOf(o0, o2), Material(o0, 'rubber'))", {'query': [{'scene_graph': [{'predicate': 'BottomQuadrant', 'variables': ['o0']}, {'predicate': 'LeftOf', 'variables': ['o0', 'o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'LeftOf', 'variables': ['o1', 'o2']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Behind', 'variables': ['o1', 'o2']}, {'predicate': 'Color', 'variables': ['o0'], 'parameter': 'gray'}, {'predicate': 'LeftOf', 'variables': ['o0', 'o2']}, {'predicate': 'Material', 'variables': ['o0'], 'parameter': 'rubber'}], 'duration_constraint': 1}]})
    # Trajectories
    test("(Behind(o0, o1), LeftQuadrant(o0), Near(o0, o1, 1.0))", {'query': [{'scene_graph': [{'predicate': 'Behind', 'variables': ['o0', 'o1']}, {'predicate': 'LeftQuadrant', 'variables': ['o0']}, {'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}], 'duration_constraint': 1}]})
    test("(BottomQuadrant(o0), Far(o0, o1, 3.0)); Near(o0, o1, 1.0)", {'query': [{'scene_graph': [{'predicate': 'BottomQuadrant', 'variables': ['o0']}, {'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3.0}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}], 'duration_constraint': 1}]})
    test("(BottomQuadrant(o0), Near(o0, o1, 1.0))", {'query': [{'scene_graph': [{'predicate': 'BottomQuadrant', 'variables': ['o0']}, {'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}], 'duration_constraint': 1}]})
    test("(Far(o0, o1, 3.0), LeftQuadrant(o0)); (LeftQuadrant(o0), Near(o0, o1, 1.0))", {'query': [{'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3.0}, {'predicate': 'LeftQuadrant', 'variables': ['o0']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'LeftQuadrant', 'variables': ['o0']}, {'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}], 'duration_constraint': 1}]})
    test("(FrontOf(o0, o1), TopQuadrant(o0))", {'query': [{'scene_graph': [{'predicate': 'FrontOf', 'variables': ['o0', 'o1']}, {'predicate': 'TopQuadrant', 'variables': ['o0']}], 'duration_constraint': 1}]})
    test("Far(o0, o1, 3.0); (Behind(o0, o1), LeftQuadrant(o0), Near(o0, o1, 1.0))", {'query': [{'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3.0}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Behind', 'variables': ['o0', 'o1']}, {'predicate': 'LeftQuadrant', 'variables': ['o0']}, {'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}], 'duration_constraint': 1}]})
    test("Far(o0, o1, 3.0); (Behind(o0, o1), Near(o0, o1, 1.0))", {'query': [{'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3.0}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Behind', 'variables': ['o0', 'o1']}, {'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}], 'duration_constraint': 1}]})
    test("Far(o0, o1, 3.0); Near(o0, o1, 1.0); Far(o0, o1, 3.0)", {'query': [{'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3.0}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3.0}], 'duration_constraint': 1}]})
    test("Near(o0, o1, 1.0); Far(o0, o1, 3.0)", {'query': [{'scene_graph': [{'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3.0}], 'duration_constraint': 1}]})
    test("Duration((FrontOf(o0, o1), LeftQuadrant(o0)), 15); Duration((LeftQuadrant(o0), RightOf(o0, o1), TopQuadrant(o0)), 5)", {'query': [{'scene_graph': [{'predicate': 'FrontOf', 'variables': ['o0', 'o1']}, {'predicate': 'LeftQuadrant', 'variables': ['o0']}], 'duration_constraint': 15}, {'scene_graph': [{'predicate': 'LeftQuadrant', 'variables': ['o0']}, {'predicate': 'RightOf', 'variables': ['o0', 'o1']}, {'predicate': 'TopQuadrant', 'variables': ['o0']}], 'duration_constraint': 5}]})
    test("Duration(Far(o0, o1, 3.0), 5); Near(o0, o1, 1.0); Far(o0, o1, 3.0)", {'query': [{'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3.0}], 'duration_constraint': 5}, {'scene_graph': [{'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Far', 'variables': ['o0', 'o1'], 'parameter': 3.0}], 'duration_constraint': 1}]})
    test("Duration(LeftOf(o0, o1), 5); (Near(o0, o1, 1.0), TopQuadrant(o0)); Duration(RightOf(o0, o1), 5)", {'query': [{'scene_graph': [{'predicate': 'LeftOf', 'variables': ['o0', 'o1']}], 'duration_constraint': 5}, {'scene_graph': [{'predicate': 'Near', 'variables': ['o0', 'o1'], 'parameter': 1.0}, {'predicate': 'TopQuadrant', 'variables': ['o0']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'RightOf', 'variables': ['o0', 'o1']}], 'duration_constraint': 5}]})
    # Warsaw
    test("(DistanceSmall(o0, o1, 100.0), Eastward2(o0), Eastward2(o1))", {'query': [{'scene_graph': [{'predicate': 'DistanceSmall', 'variables': ['o0', 'o1'], 'parameter': 100.0}, {'predicate': 'Eastward2', 'variables': ['o0']}, {'predicate': 'Eastward2', 'variables': ['o1']}], 'duration_constraint': 1}]})
    test("(Eastward2(o0), Eastward4(o1)); (Eastward2(o0), Eastward3(o1))", {'query': [{'scene_graph': [{'predicate': 'Eastward2', 'variables': ['o0']}, {'predicate': 'Eastward4', 'variables': ['o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Eastward2', 'variables': ['o0']}, {'predicate': 'Eastward3', 'variables': ['o1']}], 'duration_constraint': 1}]})
    test("(Eastward2(o0), HighAccel(o0, 2.0)); (Eastward2(o1), HighAccel(o1, 2.0))", {'query': [{'scene_graph': [{'predicate': 'Eastward2', 'variables': ['o0']}, {'predicate': 'HighAccel', 'variables': ['o0'], 'parameter': 2.0}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Eastward2', 'variables': ['o1']}, {'predicate': 'HighAccel', 'variables': ['o1'], 'parameter': 2.0}], 'duration_constraint': 1}]})
    test("(Southward1Upper(o0), Westward2(o1)); (Westward2(o0), Westward2(o1))", {'query': [{'scene_graph': [{'predicate': 'Southward1Upper', 'variables': ['o0']}, {'predicate': 'Westward2', 'variables': ['o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Westward2', 'variables': ['o0']}, {'predicate': 'Westward2', 'variables': ['o1']}], 'duration_constraint': 1}]})
    test("Duration((DistanceSmall(o0, o1, 100.0), Eastward3(o1), Eastward4(o0)), 10)", {'query': [{'scene_graph': [{'predicate': 'DistanceSmall', 'variables': ['o0', 'o1'], 'parameter': 100.0}, {'predicate': 'Eastward3', 'variables': ['o1']}, {'predicate': 'Eastward4', 'variables': ['o0']}], 'duration_constraint': 10}]})
    test("Duration((Eastward2(o0), Eastward3(o1), Faster(o0, o1, 1.5)), 5)", {'query': [{'scene_graph': [{'predicate': 'Eastward2', 'variables': ['o0']}, {'predicate': 'Eastward3', 'variables': ['o1']}, {'predicate': 'Faster', 'variables': ['o0', 'o1'], 'parameter': 1.5}], 'duration_constraint': 5}]})
    test("Duration((Eastward2(o0), Eastward4(o1)), 5); Duration((Eastward2(o0), Eastward3(o1)), 5)", {'query': [{'scene_graph': [{'predicate': 'Eastward2', 'variables': ['o0']}, {'predicate': 'Eastward4', 'variables': ['o1']}], 'duration_constraint': 5}, {'scene_graph': [{'predicate': 'Eastward2', 'variables': ['o0']}, {'predicate': 'Eastward3', 'variables': ['o1']}], 'duration_constraint': 5}]})
    # Syntax error
    test("Far(o0, o1, o2, 3)")
    test("Duration(Color(o0, 'red'), Far(o0, o1, 3), 5)")
    test("(BottomQuadrant(o0), Color_green(o0), Near_1.0(o0, o1), RightOf(o0, o1))")