# Databricks notebook source
dbutils.widgets.text("name", "", label="Please enter your name")

# COMMAND ----------

name = dbutils.widgets.get("name")

# COMMAND ----------

print(f'Hello world! {name} is comming')
