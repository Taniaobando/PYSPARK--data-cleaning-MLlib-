#Importando librerias
import time
#----------------------------------------------------------------------------#
#       Procesamiento de grandes volumenes de datos 2020-2                   #
#                     Proyecto 2 (Streaming)                                 #
#                       Alejandro Ayala Gil                                  #
#                       Esteban Cardona Gil                                  #
#                    Juan Camilo Gomez Muñoz                                 #
#                        Julian Paredes C                                    #
#                     Tania C. Obando Suárez                                 #
#----------------------------------------------------------------------------#

#Importando librerias
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql import SparkSession, DataFrameStatFunctions, DataFrameNaFunctions
from pyspark.sql.functions import *
#import seaborn as sns
#import matplotlib.pyplot as plt
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#-----------------CONOCIMIENTO Y LIMPIEZA DE LOS DATOS------------------------#

def categoricalToNumerical(df):

    #Reemplazar valores categoricos a númericos
    df = df.withColumn("school", regexp_replace("school", "GP", "0"))
    df = df.withColumn("school", regexp_replace("school", "MS", "1"))
    df = df.withColumn("sex", regexp_replace("sex", "F", "0"))
    df = df.withColumn("sex", regexp_replace("sex", "M", "1"))
    df = df.withColumn("address", regexp_replace("address", "R", "0"))
    df = df.withColumn("address", regexp_replace("address", "U", "1"))
    df = df.withColumn("famsize", regexp_replace("famsize", "LE3", "0"))
    df = df.withColumn("famsize", regexp_replace("famsize", "GT3", "1"))
    df = df.withColumn("Pstatus", regexp_replace("Pstatus", "A", "0"))
    df = df.withColumn("Pstatus", regexp_replace("Pstatus", "T", "1"))
    df = df.withColumn("Mjob", regexp_replace("Mjob", "other", "0"))
    df = df.withColumn("Mjob", regexp_replace("Mjob", "at_home", "1"))
    df = df.withColumn("Mjob", regexp_replace("Mjob", "teacher", "2"))
    df = df.withColumn("Mjob", regexp_replace("Mjob", "services", "3"))
    df = df.withColumn("Mjob", regexp_replace("Mjob", "health", "4"))
    df = df.withColumn("Fjob", regexp_replace("Fjob", "other", "0"))
    df = df.withColumn("Fjob", regexp_replace("Fjob", "at_home", "1"))
    df = df.withColumn("Fjob", regexp_replace("Fjob", "teacher", "2"))
    df = df.withColumn("Fjob", regexp_replace("Fjob", "services", "3"))
    df = df.withColumn("Fjob", regexp_replace("Fjob", "health", "4"))
    df = df.withColumn("reason", regexp_replace("reason", "other", "0"))
    df = df.withColumn("reason", regexp_replace("reason", "home", "1"))
    df = df.withColumn("reason", regexp_replace("reason", "reputation", "2"))
    df = df.withColumn("reason", regexp_replace("reason", "course", "3"))
    df = df.withColumn("guardian", regexp_replace("guardian", "father", "1"))
    df = df.withColumn("guardian", regexp_replace("guardian", "mother", "2"))
    df = df.withColumn("guardian", regexp_replace("guardian", "other", "0"))
    df = df.withColumn("schoolsup", regexp_replace("schoolsup", "no", "0"))
    df = df.withColumn("schoolsup", regexp_replace("schoolsup", "yes", "1"))
    df = df.withColumn("famsup", regexp_replace("famsup", "no", "0"))
    df = df.withColumn("famsup", regexp_replace("famsup", "yes", "1"))
    df = df.withColumn("paid", regexp_replace("paid", "no", "0"))
    df = df.withColumn("paid", regexp_replace("paid", "yes", "1"))
    df = df.withColumn("activities", regexp_replace("activities", "no", "0"))
    df = df.withColumn("activities", regexp_replace("activities", "yes", "1"))
    df = df.withColumn("nursery", regexp_replace("nursery", "no", "0"))
    df = df.withColumn("nursery", regexp_replace("nursery", "yes", "1"))
    df = df.withColumn("higher", regexp_replace("higher", "no", "0"))
    df = df.withColumn("higher", regexp_replace("higher", "yes", "1"))
    df = df.withColumn("internet", regexp_replace("internet", "no", "0"))
    df = df.withColumn("internet", regexp_replace("internet", "yes", "1"))
    df = df.withColumn("romantic", regexp_replace("romantic", "no", "0"))
    df = df.withColumn("romantic", regexp_replace("romantic", "yes", "1"))
    #df.show()
    return(df)

def stringToInt(df):

    #Casteo de todos los datos de string a int
    df = df.withColumn('school', df.school.astype("int"))
    df = df.withColumn('sex', df.sex.astype("int"))
    df = df.withColumn('age', df.age.astype("int"))
    df = df.withColumn('address', df.address.astype("int"))
    df = df.withColumn('famsize', df.famsize.astype("int"))
    df = df.withColumn('Pstatus', df.Pstatus.astype("int"))
    df = df.withColumn('Medu', df.Medu.astype("int"))
    df = df.withColumn('Fedu', df.Fedu.astype("int"))
    df = df.withColumn('Mjob', df.Mjob.astype("int"))
    df = df.withColumn('Fjob', df.Fjob.astype("int"))
    df = df.withColumn('reason', df.reason.astype("int"))
    df = df.withColumn('guardian', df.guardian.astype("int"))
    df = df.withColumn('traveltime', df.traveltime.astype("int"))
    df = df.withColumn('studytime', df.studytime.astype("int"))
    df = df.withColumn('failures', df.failures.astype("int"))
    df = df.withColumn('schoolsup', df.schoolsup.astype("int"))
    df = df.withColumn('famsup', df.famsup.astype("int"))
    df = df.withColumn('paid', df.paid.astype("int"))
    df = df.withColumn('activities', df.activities.astype("int"))
    df = df.withColumn('nursery', df.nursery.astype("int"))
    df = df.withColumn('higher', df.higher.astype("int"))
    df = df.withColumn('internet', df.internet.astype("int"))
    df = df.withColumn('romantic', df.romantic.astype("int"))
    df = df.withColumn('famrel', df.famrel.astype("int"))
    df = df.withColumn('freetime', df.freetime.astype("int"))
    df = df.withColumn('goout', df.goout.astype("int"))
    df = df.withColumn('Dalc', df.Dalc.astype("int"))
    df = df.withColumn('Walc', df.Walc.astype("int"))
    df = df.withColumn('health', df.health.astype("int"))
    df = df.withColumn('absences', df.absences.astype("int"))
    df = df.withColumn('G1', df.G1.astype("int"))
    df = df.withColumn('G2', df.G2.astype("int"))
    df = df.withColumn('G3', df.G3.astype("int"))
    #df.show()
    return(df)

def approvedOrReproved(df):
    """
    Aquí defininimos un umbral del 60% de la nota máxima para
    establecer quienes aprueban y quienes reprueban.

    Nota: Es importante hacer un casteo luego de unir la partición de los datasets,
    obtuvimos algunos errores por omitir esto.
    """

    #Estableciendo umbral para el primer periodo
    df = df.withColumn('G1', df.G1.astype("int"))
    approved = df.filter(df.G1 >= 12)
    reproved = df.filter(df.G1 < 12)
    for i in range(12):
        reproved = reproved.withColumn("G1", regexp_replace("G1", "{}".format(i), "0"))
    for i in range(12,20):
        approved = approved.withColumn("G1", regexp_replace("G1", "{}".format(i), "1"))

    df = approved.union(reproved)
    df = df.withColumn('G1', df.G1.astype("int"))

    #Estableciendo umbral para el segundo periodo
    df = df.withColumn('G2', df.G2.astype("int"))
    approved = df.filter(df.G2 >= 12)
    reproved = df.filter(df.G2 < 12)
    for i in range(12):
        reproved = reproved.withColumn("G2", regexp_replace("G2", "{}".format(i), "0"))
    for i in range(12,20):
        approved = approved.withColumn("G2", regexp_replace("G2", "{}".format(i), "1"))
    df = approved.union(reproved)
    df = df.withColumn('G2', df.G2.astype("int"))

    #Estableciendo umbral para el tercer periodo
    df = df.withColumn('G3', df.G3.astype("int"))
    approved = df.filter(df.G3 >= 12)
    reproved = df.filter(df.G3 < 12)
    for i in range(12):
        reproved = reproved.withColumn("G3", regexp_replace("G3", "{}".format(i), "0"))
    for i in range(12,20):
        approved = approved.withColumn("G3", regexp_replace("G3", "{}".format(i), "1"))
        
    #print('Número de estudiantes que rerobaron:', reproved.count())
    #print('Número de estudiantes que aprobaron:', approved.count())
    #Aquí obtuvimos 301 estudiantes reprobados y 348 estudiantes aprobados

    df = approved.union(reproved)
    df = df.withColumn('G3', df.G3.astype("int"))

    #df.count()
    #df.show()
    return(df)

#----------------------------------PREPROCESAMIENTO-----------------------------------#

def dropAtypicValues(df):
    """
    Eliminación de datos atipicos
    Nota: Para esta fase establecimos que estabamos dispuestos a eliminar hasta un 10%
    del total de los datos del dataset (649).
    """
    df=df.filter(df['age'] != 22)
    df=df.filter(df['traveltime'] != 4)
    df=df.filter(df['absences'] <17)
    df=df.filter(df['Dalc'] != 5)
    #df.count()
    #Al depurar los datos atípicos, terminamos con un total de 608 datos.
    return(df)

def dataBalancing(df):

    #Mirar balance de los datos

    approved = df.filter(df.G3 == 1)
    reproved = df.filter(df.G3 == 0)

    """
    print("cantidad final de estudiantes aprobados",approved.count())
    print("cantidad final de estudiantes reprobados",reproved.count())
    """

    #Aplicar un balanceo de los datos reduciendo la clase mayorataria
    approved=approved.sample(fraction=0.809,seed = 9403040)
    #print( "approved",approved.count())
    df = approved.union(reproved)

    print(df.dtypes,"df.dtypes_dataBalancing")

    return(df)


#----------------------------MACHINE LEARNING--------------------------------#

#--------------------VECTOR DE MÁQUINA DE SOPORTES------------------------#
#NOTA: Este modelo cuenta con los siguientes párametros predeterminados:
#      maxIter = 100
#      threshold = 0.0
#      aggregationDepth = 0.2
#      regParam = 0.0


def svm(df,trainingData,testData,maxIterValue,regParamValue,depth,thresholdValue):

    print("\n")
    print("mvs")
            
    svm = LinearSVC(labelCol="G3", featuresCol="features", maxIter=maxIterValue, regParam=regParamValue, aggregationDepth=depth, threshold=thresholdValue)

    # Fit the model
    model = svm.fit(trainingData)

    # make predictions using our trained model

    predictions = model.transform(testData)

    # estimate the accuracy of the prediction
    #Métricas de evaluación
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="G3", predictionCol="prediction", metricName="accuracy")
    accuracy = multi_evaluator.evaluate(predictions)

    multi_evaluator = multi_evaluator.setMetricName('precisionByLabel')
    precision = multi_evaluator.evaluate(predictions)

    multi_evaluator = multi_evaluator.setMetricName('f1')
    f1_score = multi_evaluator.evaluate(predictions)
    
    multi_evaluator = multi_evaluator.setMetricName('recallByLabel')
    recall = multi_evaluator.evaluate(predictions)

    bin_evaluator = BinaryClassificationEvaluator(labelCol="G3", rawPredictionCol="prediction", metricName="areaUnderROC")
    area = bin_evaluator.evaluate(predictions)
    
    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1 score = {}".format(f1_score))
    print("Area under ROC curve = {}".format(area))
    return (model)

def main():

    #Encabezado del dataframe
    headings = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob',
     'Fjob','reason','guardian','traveltime','studytime','failures','schoolsup',
     'famsup','paid','activities','nursery','higher','internet','romantic',
     'famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']
    spark = SparkSession.builder.appName("Student").getOrCreate()

    #Crear dataframe
    df=spark.read.csv('Datos streaming/read/student-por1.csv',sep=';',header=True)
    #Reemplazar valores categoricos a numericos
    df=categoricalToNumerical(df)
    #Convertir los datos de string a int
    df=stringToInt(df)
    #Convertir variables categorica a numericas
    df=approvedOrReproved(df)
    #Eliminar datos atípicos
    df=dropAtypicValues(df)
    #Balancear dataframe
    df=dataBalancing(df)

    #------------------------CREACIÓN DE LOS DATASETS FINALES---------------------#
    #Crear vectores assembler

    vector = VectorAssembler(inputCols=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob',
        'Fjob','reason','guardian','traveltime','studytime','failures','schoolsup',
        'famsup','paid','activities','nursery','higher','internet','romantic',
        'famrel','freetime','goout','Dalc','Walc','health','absences','G1'], outputCol="features")

    #Adaptar los vectores al conjunto de datos
    df_temp = vector.transform(df)

    #df_temp.show(5)
    # get dataframe with all necedf_tempssary data in the appropriate form

    final_df = df_temp.drop('school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob',
        'Fjob','reason','guardian','traveltime','studytime','failures','schoolsup',
        'famsup','paid','activities','nursery','higher','internet','romantic',
        'famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2')

    #Partición de los dataframes
    trainingData, testData= final_df.randomSplit([0.7,0.3],seed=3102020)
    mvs1_model1=svm(final_df,trainingData,testData, maxIterValue =10, thresholdValue=0.5, depth = 2, regParamValue = 0.0)
    #Finaliza la sesión de spark
    spark.stop()

#Registro de tiempo de ejecución
t1 = time.time()
main()
t2 = time.time()
print("El tiempo total es:", t2-t1)