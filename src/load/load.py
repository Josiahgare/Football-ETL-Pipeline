from src.transform import silver_transform, gold_transform

transformed_df = silver_transform.transform_data(silver_transform.df)
print(transformed_df.head(10))
