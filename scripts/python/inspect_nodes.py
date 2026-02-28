import pickle
import pandas as pd
from sklearn.tree import _tree

def get_node_rules(model_path, data_path, node_ids):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Necesitamos los nombres de las columnas para mapear los índices
    df = pd.read_csv(data_path)
    discrete_cols = [col for col in df.columns if col.startswith('d_')]
    df_encoded = pd.get_dummies(df, columns=discrete_cols, drop_first=True)
    feature_names = df_encoded.drop('target', axis=1).columns.tolist()
    
    tree_ = model.tree_
    
    def recurse(node, current_rules):
        if node in node_ids:
            print(f"\n--- REGLAS PARA EL NODO {node} ---")
            for rule in current_rules:
                print(f"  • {rule}")
            
            # Info del nodo
            n_samples = tree_.n_node_samples[node]
            values = tree_.value[node][0]
            # Si se usó class_weight='balanced', los values están pesados. 
            # Mostramos la proporción relativa para entender el riesgo.
            prob_fraud = values[1] / sum(values)
            print(f"  Resultados: Proporción de Fraude: {prob_fraud:.2%}")
            print(f"  Muestras en este nodo: {n_samples}")

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            # Izquierda (True para la condición <= threshold)
            recurse(tree_.children_left[node], current_rules + [f"{name} <= {threshold:.2f}"])
            # Derecha (False para la condición > threshold)
            recurse(tree_.children_right[node], current_rules + [f"{name} > {threshold:.2f}"])

    recurse(0, [])

if __name__ == "__main__":
    get_node_rules('model_assets/DecisionTree_Fraud_v1.pkl', 'data/fraud_sample.csv', [12, 13])
