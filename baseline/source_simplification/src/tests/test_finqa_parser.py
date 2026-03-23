from g4k.datasets.finqa.finqa_parser import FinQAParser


def test_finqa_parser():
    table = """
|    | ( in millions )                                                                            | 2011             | 2010             | 2009             |
|---:|:-------------------------------------------------------------------------------------------|:-----------------|:-----------------|:-----------------|
|  0 | sales and transfers of oil and gas produced net of production and administrative costs     | $ -7922 ( 7922 ) | $ -6330 ( 6330 ) | $ -4876 ( 4876 ) |
|  1 | net changes in prices and production and administrative costs related to future production | 12313            | 9843             | 4840             |
|  2 | extensions discoveries and improved recovery less related costs                            | 1454             | 1268             | 1399             |
|  3 | development costs incurred during the period                                               | 1899             | 2546             | 2786             |
|  4 | changes in estimated future development costs                                              | -1349 ( 1349 )   | -2153 ( 2153 )   | -3773 ( 3773 )   |
|  5 | revisions of previous quantity estimates                                                   | 2526             | 1117             | 5110             |
|  6 | net changes in purchases and sales of minerals in place                                    | 233              | -20 ( 20 )       | -159 ( 159 )     |
|  7 | accretion of discount                                                                      | 2040             | 1335             | 787              |
|  8 | net change in income taxes                                                                 | -6676 ( 6676 )   | -4231 ( 4231 )   | -4345 ( 4345 )   |
|  9 | timing and other                                                                           | 130              | 250              | -149 ( 149 )     |
| 10 | net change for the year                                                                    | 4648             | 3625             | 1620             |
| 11 | beginning of the year                                                                      | 9280             | 5655             | 4035             |
| 12 | end of year                                                                                | $ 13928          | $ 9280           | $ 5655           |
"""[1:]  # noqa: E501
    program_solution = "table_average(net change for the year, none)"
    parser = FinQAParser(table)
    result = parser.parse(program_solution)
    # just in case another interpreter has different float operations
    assert abs(float(result) - 3297.6666666666665) < 1e-8
