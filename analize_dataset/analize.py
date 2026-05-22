import pandas as pd

alto = pd.read_excel("../dataset_processed/Dataset_Index.xlsx", sheet_name="Alto_Sax_Index")
tenor = pd.read_excel("../dataset_processed/Dataset_Index.xlsx", sheet_name="Tenor_Sax_Index")

for name, df in [('ALTO', alto), ('TENOR', tenor)]:
    print(f"{name}")
    print(f"Type: {df['TYPE'].value_counts().to_dict()}")
    print(f"Measure: {df['MEASURE'].value_counts().to_dict()}")
    print(f"Metronome: {df['METRONOME_USAGE'].value_counts().to_dict()}")
    print('Tempo: ', df['TEMPO'].describe())
    
    tecnicas = ['ALTISSIMO', 'BEND', 'BREATH', 'FALL', 'FALSE FINGERING', 'GLISSANDO', 'GROWL', 'TRILL', 'VIBRATO']
    print("Techniques:")
    for tecnica in tecnicas:
        print(f'  {tecnica}: {(df[tecnica]=="YES").sum()}')