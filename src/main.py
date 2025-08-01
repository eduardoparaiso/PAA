import pandas as pd
import ast
from ecg_preprocessing import load_ECG, clean_ECG, detect_qrs, extract_beats, mean_template, cria_template
from classifier_report import classificar_com_base_nas_distancias, votacao_final, matriz_confusao, report
from dtw_utils import calcular_distancias_dtw

from utils import analisa_tempo

def aggregate_diagnostic(y_dic, agg_df):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def aggregate_subclass(y_dic, agg_df):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))

def arq_interesse(path, random_state=2025):
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Aplicando aggregate_diagnostic com agg_df como argumento
    Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))
    Y['diagnostic_subclass'] = Y.scp_codes.apply(lambda x: aggregate_subclass(x, agg_df))


    filtered_Y = Y[Y['diagnostic_subclass'].apply(lambda x: x == ['NORM'] or x == ['AMI'])]
    filtered_Y = filtered_Y[filtered_Y['validated_by_human']]

    # Cria atributo binário: 0 para NORM, 1 para MI
    filtered_Y['label'] = filtered_Y['diagnostic_subclass'].apply(lambda x: 0 if x == ['NORM'] else 1)

    # Separa as classes
    class_0 = filtered_Y[filtered_Y['label'] == 0].sample(n=440, random_state=random_state)
    class_1 = filtered_Y[filtered_Y['label'] == 1]

    # Junta os dois subconjuntos
    final_df = pd.concat([class_0, class_1]).sort_index()

    return final_df[['filename_hr', 'diagnostic_subclass','diagnostic_superclass', 'label']]



def main():
    ROOT_PATH = 'PTB-XL/'
    RANDOM_STATE = 2025

    #-------------------------------------------------------------------------------------------

    PTB_XL = arq_interesse(ROOT_PATH)
    PTB_XL_Y = PTB_XL['label']
    PTB_XL_X = PTB_XL.drop('label', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(PTB_XL_X, PTB_XL_Y,
                                                        test_size=0.1,
                                                        random_state= RANDOM_STATE)

    NORM = X_train[y_train['label'] == 0]
    AMI = X_train[y_train['label'] == 1]

    proto_norm = cria_template(df=NORM, ref_template=True)
    proto_ami = cria_template(df=AMI, ref_template=True)


    #---------- DTW ----------
    predicoes = []
    tempo_normal = tempo_ami = []
    for idx, row in X_test.iterrows():
        proto_target = cria_template(ECG_path=row['filename_hr'], canais=[6,7,8,9])
        dist_normal, tempo = calcular_distancias_dtw(proto_target, proto_norm)
        tempo_normal.append(tempo)
        dist_ami, tempo = calcular_distancias_dtw(proto_target, proto_ami)
        tempo_ami.append(tempo)
        resultado = votacao_final(classificar_com_base_nas_distancias(dist_normal, dist_ami))
        predicoes.append(resultado)

    y_dtw = y_test.copy()  # evita alteração direta se y_test for um slice
    y_dtw['predict'] = predicoes
    y_dtw['predict'] = y_dtw['predict'].replace({'NORMAL': 0, 'AMI': 1})

    matriz_confusao(y_dtw)
    analisa_tempo(tempo_normal)
    analisa_tempo(tempo_ami)
    report(y_dtw['label'], y_dtw['predict'])

    #---------- FastDTW ----------
    predicoes = []
    tempo_normal = tempo_ami = []
    for idx, row in X_test.iterrows():
        proto_target = cria_template(ECG_path=row['filename_hr'], canais=[6,7,8,9])
        dist_normal, tempo = calcular_distancias_dtw(proto_target, proto_norm, fastDTW=True)
        tempo_normal.append(tempo)
        dist_ami, tempo = calcular_distancias_dtw(proto_target, proto_ami, fastDTW=True)
        tempo_ami.append(tempo)
        resultado = votacao_final(classificar_com_base_nas_distancias(dist_normal, dist_ami))
        predicoes.append(resultado)
    
    y_fastdtw = y_test.copy()  # evita alteração direta se y_test for um slice
    y_fastdtw['predict'] = predicoes
    y_fastdtw['predict'] = y_fastdtw['predict'].replace({'NORMAL': 0, 'AMI': 1})

    matriz_confusao(y_fastdtw)
    analisa_tempo(tempo_normal)
    analisa_tempo(tempo_ami)
    report(y_fastdtw['label'], y_fastdtw['predict'])



if __name__ == '__main__':
    main()