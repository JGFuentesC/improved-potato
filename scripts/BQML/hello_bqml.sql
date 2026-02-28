-- Basado en este query, crea un pipeline para entrenar un modelo de regresión logísitica con hiperparametrización. no existe tabla de testing, por tanto debes hacer la partición tu mismo
CREATE OR REPLACE MODEL
  `calif-bot.ml_us.sme_logistic_model` OPTIONS ( MODEL_TYPE = 'LOGISTIC_REG',
    INPUT_LABEL_COLS = ['tgt'],
    DATA_SPLIT_METHOD='RANDOM', DATA_SPLIT_EVAL_FRACTION=0.2, DATA_SPLIT_TEST_FRACTION=0,
    HPARAM_TUNING_ALGORITHM='VIZIER_DEFAULT',
    AUTO_CLASS_WEIGHTS = TRUE,
    L1_REG = HPARAM_RANGE(0.01,0.1),
    L2_REG = HPARAM_RANGE(0.01,0.1),
    ENABLE_GLOBAL_EXPLAIN=TRUE,
    num_trials= 10,
    MAX_PARALLEL_TRIALS = 5,
    HPARAM_TUNING_OBJECTIVES = ['ROC_AUC']
    ) AS
SELECT
  edadEmprendedor,
  dependientesEconomicos,
  antiguedadNegocio,
  horaApertura,
  horaCierre,
  numEmpleados,
  ventasPromedioDiarias,
  sexoEmprendedor,
  escolaridadEmprendedor,
  estadoCivil,
  familiaAyuda,
  giroNegocio,
  registroVentas,
  registroContabilidad,
  CASE
    WHEN altaSAT = 's' THEN 1
    ELSE 0
END
  AS tgt
FROM
  `YOUR_PROJECT_ID.YOUR_DATASET.YOUR_TABLE`
WHERE
  altaSAT != 'o';