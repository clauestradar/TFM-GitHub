# TFM — Cosmetic Pricing and the Lipstick Index

Este repositorio contiene el análisis exploratorio (EDA) del Trabajo de Fin de Máster enfocado en el estudio de la evolución de precios de productos cosméticos y su relación con variables macroeconómicas.

## Objetivo

Analizar cómo los precios de productos cosméticos reaccionan a diferentes fases del ciclo económico, revisitando el concepto del *Lipstick Index*.

## Datos

El dataset ha sido construido a partir de:

- **Keepa**: precios históricos de productos en Amazon
- **FRED**: variables macroeconómicas (inflación, tipos de interés, desempleo)

El resultado es un panel de datos mensual con múltiples productos (ASINs) y variables económicas.

## Estructura del repositorio

- `scripts/`: scripts principales del proyecto
- `data/`: dataset final utilizado para el análisis
- `outputs/`: gráficos y resultados del EDA

## Ejecución

1. Instalar dependencias:

```bash
pip install -r requirements.txt