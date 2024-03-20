import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.preprocessing import StandardScaler


def dataset_cleaning(dataset):
    df = dataset
    df_cleaned = df.loc[:, df.isnull().mean() < .8]
    df_cleaned.drop(df_cleaned.columns[
                        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 17, 23, 25, 26, 30, 31, 32, 33, 36, 41, 48, 49, 50,
                         51, 53]], axis=1, inplace=True)
    df_cleaned.drop_duplicates(inplace=True)
    df_cleaned = df_cleaned[df_cleaned['countries_fr'].str.contains('États-Unis|France|Suisse|Allemagne|Espagne|'
                                                                    'Royaume-Uni')==True]
    df_cleaned = df_cleaned.reset_index(drop=True)

    df_cleaned[df_cleaned[['trans-fat_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g']] < 0] = np.nan

    df_cleaned.drop(df_cleaned[df_cleaned['energy_100g'] > 10000].index, inplace=True)
    df_cleaned.drop(df_cleaned[df_cleaned['fat_100g'] > 400].index, inplace=True)
    df_cleaned.drop(df_cleaned[df_cleaned['trans-fat_100g'] > 300].index, inplace=True)
    df_cleaned.drop(df_cleaned[df_cleaned['proteins_100g'] > 400].index, inplace=True)
    df_cleaned.drop(df_cleaned[df_cleaned['carbohydrates_100g'] > 2500].index, inplace=True)
    df_cleaned.drop(df_cleaned[df_cleaned['sugars_100g'] > 3000].index, inplace=True)
    features_to_manage = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'trans-fat_100g', 'carbohydrates_100g',
                          'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']

    for feature in features_to_manage:
        Q1 = df_cleaned[feature].quantile(0.25)
        Q3 = df_cleaned[feature].quantile(0.75)
        IQR = Q3 - Q1
        df_cleaned = df_cleaned[~((df_cleaned[feature] < (Q1 - 1.5 * IQR)) | (df_cleaned[feature] > (Q3 + 1.5 * IQR)))]

    return df_cleaned


def graph_pays(dataset):
    df_cleaned = dataset
    cat_pays = df_cleaned['countries_fr'].value_counts()
    big_cat_pays = cat_pays[cat_pays >= 4000]
    small_cat_pays = cat_pays[cat_pays < 4000]
    small_sums = pd.Series([small_cat_pays.sum()], index=["Autres"])
    pays_total = pd.concat([big_cat_pays, small_sums])

    fig_camembert_pays = px.pie(
        pays_total,
        values=pays_total.values,
        names=pays_total.index,
    )
    fig_distribution = px.box(df_cleaned, x="fat_100g")

    return fig_camembert_pays, fig_distribution


def cleaning_grouped_aliments(dataset):
    print("Dataset cleaning...")
    df_aliments = dataset
    df_aliments = df_aliments.astype({'pnns_groups_2':'string'})
    df_aliments = df_aliments[['pnns_groups_2', 'energy_100g', 'fat_100g', 'saturated-fat_100g',
                               'trans-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
                               'proteins_100g', 'salt_100g', 'sodium_100g']]
    df_aliments = df_aliments.groupby(by="pnns_groups_2", dropna=True).mean()
    df_aliments = df_aliments.reset_index()
    print("Cleaning complete.")

    return df_aliments


def imputation(dataset_basic, dataset_aliments):
    print("Dataset imputation...")
    df_cleaned = dataset_basic
    df_grouped_aliments = dataset_aliments
    food_category = df_grouped_aliments['pnns_groups_2'].unique()
    feature_to_complete = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'trans-fat_100g', 'carbohydrates_100g',
                           'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']

    for index, row in df_cleaned.iterrows():
        if pd.isna(row["pnns_groups_2"]):
            pass
        else:
            for feature in feature_to_complete:
                if pd.isna(row[feature]):
                    df_cleaned.at[index, feature] = df_grouped_aliments.loc[df_grouped_aliments['pnns_groups_2'] == row["pnns_groups_2"], feature].iloc[0]
                else:
                    pass

    mediane_graisse = df_cleaned['fat_100g'].median()
    mediane_graisse_sat = df_cleaned['saturated-fat_100g'].median()
    coeff_conversion_graisse = mediane_graisse / mediane_graisse_sat
    mediane_sel = df_cleaned['salt_100g'].median()
    mediane_sodium = df_cleaned['sodium_100g'].median()
    coeff_conversion_sel = mediane_sel / mediane_sodium
    mediane_carb = df_cleaned['carbohydrates_100g'].median()
    mediane_sucre = df_cleaned['sugars_100g'].median()
    mediane_energy = df_cleaned['energy_100g'].median()
    mediane_fiber = df_cleaned['fiber_100g'].median()
    mediane_protein = df_cleaned['proteins_100g'].median()
    coeff_conversion_carb = mediane_carb / mediane_sucre

    df_cleaned['saturated-fat_100g'] = df_cleaned['saturated-fat_100g'].fillna(df_cleaned['fat_100g'] / coeff_conversion_graisse)
    df_cleaned['saturated-fat_100g'] = df_cleaned['saturated-fat_100g'].fillna(mediane_graisse_sat)
    df_cleaned["fat_100g"] = df_cleaned["fat_100g"].fillna(df_cleaned['saturated-fat_100g'] * coeff_conversion_graisse)
    df_cleaned["fat_100g"] = df_cleaned["fat_100g"].fillna(mediane_graisse)
    df_cleaned['trans-fat_100g'] = df_cleaned['trans-fat_100g'].fillna(0)
    df_cleaned["sodium_100g"] = df_cleaned["sodium_100g"].fillna(df_cleaned['salt_100g'] / coeff_conversion_sel)
    df_cleaned["sodium_100g"] = df_cleaned["sodium_100g"].fillna(mediane_sodium)
    df_cleaned["salt_100g"] = df_cleaned["salt_100g"].fillna(df_cleaned['sodium_100g'] * coeff_conversion_sel)
    df_cleaned["salt_100g"] = df_cleaned["salt_100g"].fillna(mediane_sel)
    df_cleaned["sugars_100g"] = df_cleaned["sugars_100g"].fillna(df_cleaned['carbohydrates_100g'] / coeff_conversion_carb)
    df_cleaned["sugars_100g"] = df_cleaned["sugars_100g"].fillna(mediane_sucre)
    df_cleaned["carbohydrates_100g"] = df_cleaned["carbohydrates_100g"].fillna(df_cleaned['sugars_100g'] * coeff_conversion_carb)
    df_cleaned["carbohydrates_100g"] = df_cleaned["carbohydrates_100g"].fillna(mediane_carb)
    df_cleaned['energy_100g'] = df_cleaned['energy_100g'].fillna(mediane_energy)
    df_cleaned['fiber_100g'] = df_cleaned['fiber_100g'].fillna(mediane_fiber)
    df_cleaned['proteins_100g'] = df_cleaned['proteins_100g'].fillna(mediane_protein)

    return df_cleaned


def nutrition_grade_to_numeric(value):
    if value == "a":
        return 0
    elif value == "b":
        return 1
    elif value == "c":
        return 2
    elif value == "d":
        return 3
    elif value == "e":
        return 4


def nutrition_grade_to_string(value):
    if value == 0:
        return "A"
    elif value == 1:
        return "B"
    elif value == 2:
        return "C"
    elif value == 3:
        return "D"
    elif value == 4:
        return "E"


def complete_nutrition_grade(dataset):
    df_cleaned = dataset
    df_knn = df_cleaned[['energy_100g', 'fat_100g', 'saturated-fat_100g', 'trans-fat_100g', 'carbohydrates_100g',
                         'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g',
                         'nutrition_grade_numeric']]
    df_knn_with_target = df_knn.dropna(subset=['nutrition_grade_numeric'])
    df_knn_incomplete = df_knn[df_knn['nutrition_grade_numeric'].isna()]

    scaler = StandardScaler()
    knn = pickle.load(open("model_knn_nutrition_grade", 'rb'))

    scaler.fit(df_knn_incomplete.drop('nutrition_grade_numeric', axis=1))
    scaled_features_to_predict = scaler.transform(df_knn_incomplete.drop('nutrition_grade_numeric', axis=1))
    df_training_to_predict = pd.DataFrame(scaled_features_to_predict, columns=df_knn_incomplete.columns[:-1])
    final_pred = knn.predict(df_training_to_predict)
    df_knn_incomplete['nutrition_grade_numeric'] = final_pred

    df_complete = pd.concat([df_knn_with_target, df_knn_incomplete])
    df_complete = df_complete.sort_index()
    df_finale = pd.merge(df_cleaned[['product_name', 'brands', 'countries_fr', 'pnns_groups_2', 'ingredients_text',
                                     'additives_n', 'additives']], df_complete, left_index=True, right_index=True)
    df_finale = df_finale.dropna(subset=['product_name', 'brands'])
    df_finale = df_finale.reset_index(drop=True)
    print("Imputation complete.")

    return df_finale
