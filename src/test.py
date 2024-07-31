# Databricks notebook source
name = dbutils.widgets.get("name")

# COMMAND ----------

print(f'Hello world! {name} is comming')
