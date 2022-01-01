import pandas as pd
import os
import matplotlib.pyplot as plt


class IMUDataset:
    def __init__(self, df=None, path=None):
        self.df = df
        self.path = path
        # self.df.dropna(axis=1, inplace=True)
        self.header = None
    
    def grab_imu_header(self):
        new_head = {}  # Initialize dict of new header values and number of associated columns
        count = 0  # Initialize associated columns count
        first = True  # Indicates first column

        # Loop through first row and test whether value is numeric. Append to new_head if non-numeric
        for value, col in zip(self.df.iloc[0], self.df.columns):
            try:
                int(value)
                count += 1  # Increase associated columns count

                # If value is last, append latest exception
                if value == self.df.iloc[0].tolist()[-1]:
                    new_head[future_col] = count

            except:
                print(self.df.columns)
                self.df.drop(columns=[col], axis=1, inplace=True)
                if not first:  # Avoid first because no future_col stored
                    new_head[future_col] = count
                else:
                    first = False
                count = 0  # Reset count
                future_col = value  # Future == current
        
        return new_head

    def header_from_dict(self, head_dict):
        new_head = []

        for key, value in head_dict.items():
            [new_head.append('{0}_{1}'.format(key.split(':')[0], i)) for i in range(value)]
        
        self.df.columns = new_head
        
        return new_head
        
    def read_single_file(self, path):
        df = pd.read_excel(path, header=None)
        return df.dropna(axis=1)
        
    def multi_concat(self, plot_features=False, plot_label=False):
        
        concat_df = pd.DataFrame()
        for file in os.listdir(self.path):
            df = self.read_single_file(os.path.join(self.path, file))
            concat_df = pd.concat((concat_df, df))
            
            # if plot_features and plot_label:
            #     features = df.drop(plot_label)
            #     plt.plot(df)
            # elif plot_features or plot_label:
            #     raise Exception('''To individually plot concatenated features both plot features and label parameters 
            #                     must be defined''')
            
        self.df = concat_df
        print(concat_df.columns)
        
    def smooth_outliers(self):
        stds_df = self.df.std()
        means_df = self.df.mean()
        for col_n, col_name in enumerate(self.df.columns):
            
            for row_n, value in enumerate(self.df[col_name]):
                
                if value < means_df[col_n] - 3*stds_df[col_n] or value > means_df[col_n] + 3*stds_df[col_n]:
                    # Look at adjacent values +/- 10 excluding the value of interest
                    smoothed_val = (self.df[col_name].iloc[row_n-10:row_n] + self.df[col_name].iloc[row_n:row_n+10]) / 2
                    self.df[col_name].iloc[row_n] = smoothed_val
                    
    def plot_each_feature(self, times=['Time_0', 'Time_1'], label='Angle_0'):
        for file in os.listdir(self.path):
            # df = self.read_single_file(os.path.join(self.path, file))
            df = pd.read_excel(os.path.join(self.path, file))
            single_imu = IMUDataset(df=df)
            header = single_imu.grab_imu_header()
            single_imu.header_from_dict(header)
            
            df = single_imu.df
            cols = times + [label]
            df.drop(labels=cols, inplace=True, axis=1)
            
            
            plt.plot(range(len(df)), df)
            plt.title(file)
            plt.legend(df.columns)
            plt.show()
    

if __name__ == '__main__':
    path = 'ml_data_v2'
    # path = 'train_data/Keller_Emily_Walking4.xlsx'
    # data = pd.read_excel(path, header=None)
    
    # imu = IMUDataset(path=path)
    # imu.multi_concat()
    # header = imu.grab_imu_header()
    # imu.header_from_dict(header)
    # imu.df.to_excel('merged_data.xlsx', index=False)
    
    imu = IMUDataset(path='ml_data_v2')
    imu.plot_each_feature()
    