from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
from functions import *

df = pd.read_csv("fr.openfoodfacts.org.products.csv", sep="\t")
df_cleaned = dataset_cleaning(df)

fig_camembert_pays, fig_distribution = graph_pays(df_cleaned)

df_aliments = cleaning_grouped_aliments(df_cleaned)

df_cleaned = imputation(df_cleaned, df_aliments)
df_cleaned['nutrition_grade_numeric'] = df_cleaned['nutrition_grade_fr'].map(nutrition_grade_to_numeric)
df_finale = complete_nutrition_grade(df_cleaned)
df_aliments.rename(columns={"pnns_groups_2": "catégorie d'aliments"}, inplace=True)


df_table = df_finale
df_table['index'] = range(1, len(df_table) + 1)

df_table['nutrition_grade_string'] = df_table['nutrition_grade_numeric'].map(nutrition_grade_to_string)

df_table = df_table[['index', 'product_name', 'brands', 'countries_fr', 'additives_n', 'pnns_groups_2', 'energy_100g',
                     'fat_100g', 'saturated-fat_100g', 'trans-fat_100g', 'carbohydrates_100g', 'sugars_100g',
                     'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g', 'nutrition_grade_string']]
df_table = df_table.round({'energy_100g': 2, 'fat_100g': 2, 'saturated-fat_100g': 2, 'trans-fat_100g': 2, 'carbohydrates_100g': 2,
                'sugars_100g': 2, 'fiber_100g': 2, 'proteins_100g': 2, 'salt_100g': 2, 'sodium_100g': 2})
df_table.rename(columns={"product_name": "nom du produit", "brands": "marque", "countries_fr": "pays",
                         "pnns_groups_2": "catégorie d'aliments", "nutrition_grade_string": "grade nutritionnel"},
                inplace=True)


# Création de la page web
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

PAGE_SIZE = 10

# Layout de la page
app.layout = dbc.Container(
    [
        dcc.Store(id='store'),
        html.Div(children=[
            html.Div([
                html.H1(['OPEN FOOD DASHBOARD'],
                        style={'marginLeft': 60, 'marginRight': 0, 'marginTop': 60, 'marginBottom': 40,
                               'padding': '6px 0px 0px 8px'}
                        ),
                html.Div(['Explorez plus de 200000 aliments différents accompagnés de leurs valeurs nutritionnelles '
                            'afin de vous aider à mieux manger au quotidien. Les données utilisées sont celles de'
                          ' https://world.openfoodfacts.org/ .'],
                         style={'marginLeft': 0, 'marginRight': 0, 'marginTop': 0, 'marginBottom': 50,
                                'padding': '6px 0px 0px 8px'}
                         )
                ]),

            html.Hr(),

            html.Div([
                html.H3('Origine des aliments'),
                html.Div('Pie chart de la répartition par pays des aliments présent dans la base de données.'),
                dcc.Graph(id='pie_chart_pays', figure=fig_camembert_pays)]),

            html.Div([
                html.H3("Distribution des valeurs nutritionnelles"),
                html.Div(['Représentation des valeurs nutritionnelles présentes dans nos données sous la forme de box plots.'],
                         # style={'marginLeft': 10, 'marginRight': 10, 'marginTop': 10, 'marginBottom': 10,
                         #        'backgroundColor': '#F7FBFE',
                         #        'border': 'thin lightgrey dashed', 'padding': '6px 0px 0px 8px'}
                         ),

                html.Div([
                    dcc.Dropdown(id='nutritional_var',
                                 value='energy_100g',
                                 options=['energy_100g', 'fat_100g', 'saturated-fat_100g',
                                          'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
                                          'salt_100g', 'sodium_100g']
                                 )],
                    style={'marginLeft': 10, 'marginRight': 800, 'marginTop': 20, 'marginBottom': 4,
                           'padding': '6px 0px 0px 8px'}
                    ),

                dcc.Graph(id='graph_distrib_nutrition')]),

            html.Div([
                html.H3("Valeurs nutritionnelles par groupes d'aliments"),
                html.Div('Bar plot des valeurs nutritionnelles moyennes des aliments de la base de données '
                                  'selon leur catégorie.'),
                html.Div([
                    dcc.Dropdown(id='nutritional_var_2',
                                 value='sugars_100g',
                                 options=['energy_100g', 'fat_100g', 'saturated-fat_100g',
                                          'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
                                          'salt_100g', 'sodium_100g']
                                 )],
                    style={'marginLeft': 10, 'marginRight': 800, 'marginTop': 20, 'marginBottom': 4,
                           'padding': '6px 0px 0px 8px'}
                    ),
                dcc.Graph(id='graph_group_aliments_nutrition',
                          style={'width': '70vw', 'height': '50vh'}
                          ),
                html.Hr(),
                ]),

            html.Div([
                html.H2("Explorez les aliments"),
                html.Div(["Utilisez les filtres et réorganisez les colonnes afin d'explorer les aliments disponibles. "
                            "Les lignes sont colorisés en fonction du grade nutritionnel de l'aliment. "],
                            style={'marginLeft': 0, 'marginRight': 0, 'marginTop': 0, 'marginBottom': 20,
                                                       'padding': '6px 0px 0px 8px'}
                        ),
                dash_table.DataTable(
                    id='datatable-paging',
                    columns=[{"name": i, "id": i}
                             for i in df_table.columns],
                    page_current=0,
                    page_size=PAGE_SIZE,
                    page_action='custom',
                    sort_action='custom',
                    filter_action="custom",
                    filter_query='',
                    sort_mode='multi',
                    sort_by=[],
                    style_cell=dict(textAlign='left'),
                    style_header=dict(backgroundColor="paleturquoise"),
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{grade nutritionnel} = A'},
                            'backgroundColor': '#1a741a',
                            'color': 'white'
                        },
                        {
                            'if': {'filter_query': '{grade nutritionnel} = B'},
                            'backgroundColor': '#7db830',
                            'color': 'black'
                        },
                        {
                            'if': {'filter_query': '{grade nutritionnel} = C'},
                            'backgroundColor': '#f0c713',
                            'color': 'black'
                        },
                        {
                            'if': {'filter_query': '{grade nutritionnel} = D'},
                            'backgroundColor': '#e8742c',
                            'color': 'white'
                        },
                        {
                            'if': {'filter_query': '{grade nutritionnel} = E'},
                            'backgroundColor': '#cb0003',
                            'color': 'white'
                        },
                    ]
                )
            ]),
        ])
    ])


# Define callback to update graph
@app.callback(
    Output('graph_distrib_nutrition', 'figure'),
    Output('graph_group_aliments_nutrition', 'figure'),
    Input("nutritional_var", "value"),
    Input("nutritional_var_2", "value"))
def update_figure(nutritional_var, nutritional_var_2):
    distrib_figure = px.box(df_cleaned, x=nutritional_var)
    group_aliments_figure = px.bar(df_aliments, x="catégorie d'aliments", y=nutritional_var_2, color="catégorie d'aliments").update_xaxes(categoryorder="total ascending")

    return distrib_figure, group_aliments_figure


operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


@app.callback(
    Output('datatable-paging', 'data'),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"),
    Input('datatable-paging', "sort_by"),
    Input('datatable-paging', "filter_query"))
def update_table(page_current, page_size, sort_by, filter):
    filtering_expressions = filter.split(' && ')
    print(sort_by)
    if len(sort_by):
        dff = df_table.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )

        for filter_part in filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)

            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                # this is a simplification of the front-end filtering logic,
                # only works with complete fields in standard format
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]



    else:
        # No sort is applied
        dff = df_table

        for filter_part in filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)

            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                # this is a simplification of the front-end filtering logic,
                # only works with complete fields in standard format
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    return dff.iloc[
        page_current*page_size:(page_current + 1)*page_size
    ].to_dict('records')


if __name__ == '__main__':
    app.run(debug=True)
