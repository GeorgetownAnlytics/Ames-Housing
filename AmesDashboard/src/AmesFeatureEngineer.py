import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

class AmesFeatureEngineer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.columns_to_drop = ['Garage_Yr_Blt', "Condition_2", "Utilities", "Roof_Matl", "Latitude", "Longitude"]
        # Add other necessary initialization parameters

    def _drop_columns(self, df):
        df.drop(self.columns_to_drop, axis=1, inplace=True)

    def _encode_and_bin(self, df):
        # Replace categories with numbers
        # Example for 'Alley': {"No_Alley_Access":0, "Gravel":1, "Paved":2}
        replacements = {
            # Add all your replacements here following the example format
            "Bsmt_Cond" : {
                "No_Basement" : 0, "Poor" : 1, "Fair" : 2, "Typical": 3, "Good" : 4, "Excellent" : 5},
                "Bsmt_Exposure" : {"No_Basement":0,"No" : 1, "Mn" : 2, "Av": 3, "Gd" : 4},
                "BsmtFin_Type_1" : {"No_Basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
                "BsmtFin_Type_2" : {"No_Basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
                "Alley": {"No_Alley_Access":0, "Gravel":1, "Paved":2},
                "Street": {"Grvl":0, "Pave":1},
                "Central_Air":{"Y":1, "N":0}, 
                "Land_Contour": {"Bnk":0,"Lvl":1,"HLS":2, "Low":3},
                "Bldg_Type":{"OneFam":0,"Duplex":1,"TwnhsE":2,"Twnhs":3,"TwoFmCon":4},
                "Bsmt_Qual" : {"No_Basement" : 0, "Poor" : 1, "Fair" : 2, "Typical": 3, "Good" : 4, "Excellent" : 5},
                "Exter_Cond" : {"Poor" : 1, "Fair" : 2, "Typical": 3, "Good": 4, "Excellent" : 5},
                "Exter_Qual" : {"Poor" : 1, "Fair" : 2, "Typical": 3, "Good": 4, "Excellent" : 5},
                "Fireplace_Qu" : {"No_Fireplace" : 0, "Poor" : 1, "Fair" : 2, "Typical" : 3, "Good" : 4, "Excellent" : 5},
                "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},
                "Garage_Qual" : {"No_Garage" : 0, "Poor" : 1, "Fair" : 2, "Typical" : 3, "Good" : 4, "Excellent" : 5},
                "Heating_QC" : {"Poor" : 1, "Fair" : 2, "Typical": 3, "Good": 4, "Excellent" : 5},
                "Garage_Finish":{"No_Garage":0, "Unf":1, "RFn":2, "Fin":3},
                "Kitchen_Qual" : {"Poor" : 1, "Fair" : 2, "Typical": 3, "Good": 4, "Excellent" : 5},
                "Land_Slope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                "Lot_Shape" : {"Irregular" : 1, "Moderately_Irregular" : 2, "Slightly_Irregular" : 3, "Regular" : 4},
                "Paved_Drive" : {"Dirt_Gravel" : 0, "Paved" : 1, "Partial_Pavement" : 2},
                "Pool_QC" : {"No_Pool" : 0, "Fair" : 1, "Typical" : 2, "Good" : 3, "Excellent" : 4},
                "Overall_Qual":{"Very_Excellent": 10, "Excellent":9,"Very_Good":8, "Good":7, "Above_Average":6, "Average": 5, "Below_Average":4, "Fair": 3,"Poor":2, "Very_Poor":1},
                "Overall_Cond" :{"Very_Excellent": 10, "Excellent":9,"Very_Good":8, "Good":7, "Above_Average":6, "Average": 5, "Below_Average":4, "Fair": 3,"Poor":2, "Very_Poor":1},
                "Electrical ": {"Unknown": 0,"SBrkr ": 1,"FuseA": 2, "FuseF": 3, "FuseP": 4,"Mix": 5 },
                "Fence":{"No_Fence":0, "Minimum_Wood_Wire":1,"Good_Wood":2,"Minimum_Privacy":3,"Good_Privacy":4},
                "Garage_Cond" : {"No_Garage" : 0, "Poor" : 1, "Fair" : 2, "Typical" : 3, "Good" : 4, "Excellent" : 5}
        }
        df.replace(replacements, inplace=True)

        # Binning Year_Built as an example
        Age_range = df['Year_Built'].max() - df['Year_Built'].min()
        min_value = int(np.floor(df['Year_Built'].min()))
        max_value = int(np.ceil(df['Year_Built'].max()))
        inter_value = int(np.round(Age_range / 9))
        intervals = [i for i in range(min_value, max_value + inter_value, inter_value)]
        labels = [i for i in range(1, len(intervals))]
        df['Year_Built'] = pd.cut(df['Year_Built'], bins=intervals, labels=labels, include_lowest=True)
        df['Year_Built'] = df['Year_Built'].astype('int64', copy=False, errors="ignore")

        # ... apply other binning logic in a similar way ...


    def _one_hot_encode(self, df):
        return pd.get_dummies(df)

    def _clip_outliers(self, df, variables):
        for var in variables:
            IQR = df[[var]].quantile(0.75) - df[[var]].quantile(0.25)
            Lower_fence = float(df[[var]].quantile(0.25) - (IQR * 3))
            Upper_fence = float(df[[var]].quantile(0.75) + (IQR * 3))
            df[var].clip(Lower_fence, Upper_fence, inplace=True)
        return df

    def _remove_correlated_and_constant_features(self, df, corr_threshold=0.7):
        # Removing correlated features
        corr_matrix = df.corr()
        correlated_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                    colname = corr_matrix.columns[i]
                    correlated_features.add(colname)
        df.drop(labels=correlated_features, axis=1, inplace=True)

        # Removing constant features
        constant_features = [feat for feat in df.columns if df[feat].std() <= 0.1]
        df.drop(labels=constant_features, axis=1, inplace=True)


    def fit_transform(self, df):
        self._drop_columns(df)
        self._encode_and_bin(df)
        self._one_hot_encode(df)
        # Add other transformations
        self._clip_outliers(df, ['continous_var1', 'continous_var2']) # Replace with actual variable names
        self._remove_correlated_and_constant_features(df)
        
        # Fit and transform scaling
        df_scaled = df.drop(['Sale_Price'], axis=1)
        self.scaler.fit(df_scaled)
        df_scaled = self.scaler.transform(df_scaled)
        
        return df_scaled

    def transform(self, df):
        self._drop_columns(df)
        self._encode_and_bin(df)
        self._one_hot_encode(df)
        # Add other transformations
        self._clip_outliers(df, ['continous_var1', 'continous_var2']) # Replace with actual variable names
        self._remove_correlated_and_constant_features(df)
        
        # Transform scaling
        df_scaled = df.drop(['Sale_Price'], axis=1)
        df_scaled = self.scaler.transform(df_scaled)
        
        return df_scaled

# Usage:
# engineer = AmesFeatureEngineer()
# X_train_scaled = engineer.fit_transform(train_df)
# X_test_scaled = engineer.transform(test_df)
