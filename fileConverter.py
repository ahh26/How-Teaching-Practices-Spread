import pandas as pd
def txt_to_dataframe(filepath):
    df=pd.read_csv(filepath,delimiter='|')
    return df

def txt_to_csv(input_filepath, output_filepath):
    df=txt_to_dataframe(input_filepath)
    df.to_csv(output_filepath, index=False)
    

filepath='/Users/antheaaaa_hax/Desktop/EducationalResourcesDistributionProject/enrolment_by_grade_2023-2024_en.txt'
output_csv='/Users/antheaaaa_hax/Desktop/EducationalResourcesDistributionProject/enrolment_by_grade_2023-2024_en.csv'
df=txt_to_dataframe(filepath)
print(df.columns)
print(df.shape)

txt_to_csv(filepath,output_csv)
        