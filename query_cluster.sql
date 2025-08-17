WITH clientes AS (
	SELECT DISTINCT 
  		cliente_id_original AS _raw_cliente_id, dt_ini_compra
  	FROM 
  		schema.table_compras
  	WHERE marca_id = :MARCA_ID
    	AND EXTRACT(YEAR FROM dt_ref_compra) = 2024
),
dado_cliente AS (
  	SELECT
    	cliente_id_original,
    	sexo_cliente,
    	TRIM(uf_cliente) AS uf,
    	ROW_NUMBER() OVER(PARTITION BY cliente_id_original ORDER BY cliente_id_original) rn
  	FROM 
  		schema.table_clientes -- <SUBSTITUIR: schema.table_clientes>
  	GROUP BY 
  		cliente_id_original, sexo_cliente, uf_cliente
),
compra_2024 AS (
  	SELECT
    	cliente_id_original,
    	SUM(vl_liq_compra) AS total_compra,
    	SUM(vl_brt_pdv) - SUM(vl_liq_compra) AS desconto,
    	SUM(CASE WHEN produto_cod = :PRODUCT_CODE_30 THEN 1 ELSE 0 END) AS quant_30,
    	SUM(CASE WHEN produto_cod = :PRODUCT_CODE_60 THEN 1 ELSE 0 END) AS quant_60,
    	SUM(CASE WHEN produto_cod = :PRODUCT_CODE_8  THEN 1 ELSE 0 END) AS quant_8
  	FROM 
  		schema.table_compras
  	WHERE EXTRACT(YEAR FROM dt_ref_compra) = 2024
    	AND marca_id = :MARCA_ID
  	GROUP BY 
  		cliente_id_original
)
SELECT
	clientes.cliente_id_original,
  	COUNT(*) AS nr_compras,
  	ROUND(compra_2024.total_compra / COUNT(*), 2) AS ticket_medio,
  	dado_cliente.sexo_cliente AS sexo,
  	dado_cliente.uf AS estado,
  	compra_2024.desconto,
  	compra_2024.quant_30,
  	compra_2024.quant_60,
  	compra_2024.quant_8
FROM 
	clientes
LEFT JOIN
	compra_2024 ON clientes.cliente_id_original = compra_2024.cliente_id_original
LEFT JOIN
	dado_cliente ON clientes.cliente_id_original = dado_cliente.cliente_id_original
WHERE dado_cliente.rn = 1
GROUP BY
  clientes.cliente_id_original,
  compra_2024.total_compra,
  sexo,
  estado,
  desconto,
  compra_2024.quant_30,
  compra_2024.quant_60,
  compra_2024.quant_8
ORDER BY 
	nr_compras DESC;
