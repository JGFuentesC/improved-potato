WITH
  agg_proj AS (
    -- Agregación inicial por proyecto y minuto
    SELECT 
      ts, 
      project, 
      SUM(downloads) AS downloads
    FROM `{BQ_TABLE}`
    GROUP BY ts, project
  ),
  
  growth_base AS (
    -- Cálculo del crecimiento minuto a minuto (Aceleración instantánea)
    -- Fórmula: (V_actual - V_anterior) / V_anterior
    SELECT 
      *,
      SAFE_DIVIDE(
        downloads - LAG(downloads) OVER (PARTITION BY project ORDER BY ts),
        LAG(downloads) OVER (PARTITION BY project ORDER BY ts)
      ) AS instant_growth
    FROM agg_proj
  ),

  final_features AS (
    SELECT
      ts,
      project,
      downloads,
      
      -- VENTANA 1-5 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS x_avg_dwn_1_5,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS x_std_dwn_1_5,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS x_avg_growth_1_5,

      -- VENTANA 6-10 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 10 PRECEDING AND 6 PRECEDING) AS x_avg_dwn_6_10,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 10 PRECEDING AND 6 PRECEDING) AS x_std_dwn_6_10,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 10 PRECEDING AND 6 PRECEDING) AS x_avg_growth_6_10,

      -- VENTANA 11-15 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 15 PRECEDING AND 11 PRECEDING) AS x_avg_dwn_11_15,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 15 PRECEDING AND 11 PRECEDING) AS x_std_dwn_11_15,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 15 PRECEDING AND 11 PRECEDING) AS x_avg_growth_11_15,

      -- VENTANA 16-20 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 20 PRECEDING AND 16 PRECEDING) AS x_avg_dwn_16_20,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 20 PRECEDING AND 16 PRECEDING) AS x_std_dwn_16_20,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 20 PRECEDING AND 16 PRECEDING) AS x_avg_growth_16_20,

      -- VENTANA 21-25 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 25 PRECEDING AND 21 PRECEDING) AS x_avg_dwn_21_25,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 25 PRECEDING AND 21 PRECEDING) AS x_std_dwn_21_25,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 25 PRECEDING AND 21 PRECEDING) AS x_avg_growth_21_25,

      -- VENTANA 26-30 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 30 PRECEDING AND 26 PRECEDING) AS x_avg_dwn_26_30,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 30 PRECEDING AND 26 PRECEDING) AS x_std_dwn_26_30,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 30 PRECEDING AND 26 PRECEDING) AS x_avg_growth_26_30,

      -- VENTANA 31-35 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 35 PRECEDING AND 31 PRECEDING) AS x_avg_dwn_31_35,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 35 PRECEDING AND 31 PRECEDING) AS x_std_dwn_31_35,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 35 PRECEDING AND 31 PRECEDING) AS x_avg_growth_31_35,

      -- VENTANA 36-40 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 40 PRECEDING AND 36 PRECEDING) AS x_avg_dwn_36_40,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 40 PRECEDING AND 36 PRECEDING) AS x_std_dwn_36_40,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 40 PRECEDING AND 36 PRECEDING) AS x_avg_growth_36_40,

      -- VENTANA 41-45 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 45 PRECEDING AND 41 PRECEDING) AS x_avg_dwn_41_45,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 45 PRECEDING AND 41 PRECEDING) AS x_std_dwn_41_45,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 45 PRECEDING AND 41 PRECEDING) AS x_avg_growth_41_45,

      -- VENTANA 46-50 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 50 PRECEDING AND 46 PRECEDING) AS x_avg_dwn_46_50,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 50 PRECEDING AND 46 PRECEDING) AS x_std_dwn_46_50,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 50 PRECEDING AND 46 PRECEDING) AS x_avg_growth_46_50,

      -- VENTANA 51-55 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 55 PRECEDING AND 51 PRECEDING) AS x_avg_dwn_51_55,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 55 PRECEDING AND 51 PRECEDING) AS x_std_dwn_51_55,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 55 PRECEDING AND 51 PRECEDING) AS x_avg_growth_51_55,

      -- VENTANA 56-60 MINUTOS
      AVG(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 60 PRECEDING AND 56 PRECEDING) AS x_avg_dwn_56_60,
      STDDEV(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 60 PRECEDING AND 56 PRECEDING) AS x_std_dwn_56_60,
      AVG(instant_growth) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 60 PRECEDING AND 56 PRECEDING) AS x_avg_growth_56_60,

      -- VARIABLE OBJETIVO: Suma de descargas en los SIGUIENTES 10 minutos
      SUM(downloads) OVER (PARTITION BY project ORDER BY ts ROWS BETWEEN 1 FOLLOWING AND 10 FOLLOWING) AS tgt,
      
      -- Identificador de fila para limpieza
      ROW_NUMBER() OVER (PARTITION BY project ORDER BY ts) AS rn
    FROM growth_base
  )

SELECT * FROM final_features 
WHERE rn > 60 -- Eliminamos el "frío" inicial (donde las ventanas no tienen datos suficientes)
  AND tgt IS NOT NULL -- Eliminamos el final (donde no hay futuro para predecir)
ORDER BY ts, project;