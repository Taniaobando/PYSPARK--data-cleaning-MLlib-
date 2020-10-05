#----------------------------------------------------------------------------#
#       Procesamiento de grandes volumenes de datos 2020-2                   #
#               Proyecto 1 (data cleaning + MLlib)                           #
#                       Alejandro Ayala Gil                                  #
#                       Esteban Cardona Gil                                  #
#                    Juan Camilo Gomez Muñoz                                 #
#                        Julian Paredes C                                    #
#                    Tania C. Obando Suárez                                  #
#----------------------------------------------------------------------------#

#Importando librerias
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql import SparkSession, DataFrameStatFunctions, DataFrameNaFunctions
from pyspark.sql.functions import *
#import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


#------------------------FUNCIONES AUXILIARES----------------------------------#

def correlation(dataframe,headings):
    n = len(headings)
    corr = [[1 for _ in range(n)] for _ in range(n)]
    dic=dict()
    for i in range(n):
        for j in range(i+1,n):
            corr[i][j] = dataframe.corr(headings[i],headings[j])
            corr[j][i] = corr[i][j]
            if corr[i][j]>0.70:
                dic[(headings[i],headings[j])]=corr[i][j]
    print(dic)
    
    #Analizar correlaciones a partir del mapa de calor de correlaciones

    """
    #Correlaciones positivamente fuertes
    print('Correlación entre G1 y G2:', df.corr('G1','G2'))
    print('Correlación entre G1 y G3:', df.corr('G1','G3'))
    print('Correlación entre G2 y G3:', df.corr('G2','G3'))

    #Correlaciones positivamente moderadas
    print('Correlación entre Medu y Fedu:', df.corr('Medu','Fedu'))
    print('Correlación entre Walc y Dalc:', df.corr('Walc','Dalc'))

    #Correlaciones negativamente moderadas
    print('Correlación entre school y address:', df.corr('school','address'))
    print('Correlación entre traveltime y address:', df.corr('traveltime','address'))
    print('Correlación entre failures y G1:', df.corr('failures','G1'))
    print('Correlación entre failures y G2:', df.corr('failures','G2'))
    print('Correlación entre failures y G3:', df.corr('failures','G3'))
    """          
    return corr

#-----------------CONOCIMIENTO Y LIMPIEZA DE LOS DATOS------------------------#

def  nullData(df):
    #Datos nulos
    total_null = df.filter("school is null").count() + df.filter("sex is null").count() + df.filter("age is null").count()
    total_null+= df.filter("address is null").count() + df.filter("famsize is null").count() + df.filter("Pstatus is null").count()
    total_null+= df.filter("Medu is null").count() + df.filter("Fedu is null").count() + df.filter("Mjob is null").count()
    total_null+= df.filter("Fjob is null").count() + df.filter("reason is null").count() + df.filter("guardian is null").count()
    total_null+= df.filter("traveltime is null").count() + df.filter("studytime is null").count() + df.filter("failures is null").count()
    total_null+= df.filter("schoolsup is null").count() + df.filter("famsup is null").count() + df.filter("paid is null").count()
    total_null+= df.filter("activities is null").count() + df.filter("nursery is null").count() + df.filter("higher is null").count()
    total_null+= df.filter("internet is null").count() + df.filter("romantic is null").count() + df.filter("famrel is null").count()
    total_null+= df.filter("freetime is null").count() + df.filter("goout is null").count() + df.filter("Dalc is null").count()
    total_null+= df.filter("Walc is null").count() + df.filter("health is null").count() + df.filter("absences is null").count()
    total_null+= df.filter("G1 is null").count() + df.filter("G2 is null").count() + df.filter("G3 is null").count()
    print(total_null,"total_null")
    return(total_null)
    #Encontramos que el dataframe inicial no tiene datos faltantes o datos nulos.

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

#----------------------------------ANALISIS-----------------------------------#

#Visualización de las medidas de centralidad

def describeData(df):
    #información estadistica acerca de los datos 
    df.describe().toPandas()
    df.toPandas().mode()

def boxWhiskerPlot(df,headings):
    #Diagramas de cajas y bigotes 
    for x in headings:
        print(x)
        plt.boxplot(df.toPandas()[x],vert = 0)
        plt.show()

def countAtypicValues(df):
    #Analisis de los diagramas 
    #cantidad de datos atipicos

    atypic_age_22=df.filter(df['age'] == 22).count()
    atypic_p_status_0=df.filter(df['Pstatus'] == 0).count()#viven padres juntos o separados 
    atypic_travel_time_4=df.filter(df['traveltime'] == 4).count()#tiempo de la casa a el colegio
    atypic_studytime_4=df.filter(df['studytime'] == 4).count()#tiempo de estudio
    atypic_failures_1=df.filter(df['failures'] == 1).count()#número de fallos de clases anteriores
    atypic_failures_2=df.filter(df['failures'] == 2).count()
    atypic_failures_3=df.filter(df['failures'] == 3).count() 
    atypic_schoolsup_1=df.filter(df['schoolsup'] == 1).count()#apoyo educativo adicional
    atypic_paid_1=df.filter(df['paid'] == 1).count()#clases extra pagadas dentro de la asignatura del curso (portugués)
    atypic_nursery_0=df.filter(df['nursery'] == 0).count()#asistio a la guarderia
    atypic_higher_0=df.filter(df['higher'] == 0).count()#piensa  cursar estudios superiores
    atypic_internet_0=df.filter(df['internet'] == 0).count()
    atypic_famrel_1=df.filter(df['famrel'] == 1).count()#calidad de las relaciones familiares
    atypic_famrel_2=df.filter(df['famrel'] == 2).count()
    atypic_freetime_1=df.filter(df['freetime'] == 1).count()#tiempo libre despues de la escuela
    atypic_Dalc_4=df.filter(df['Dalc'] == 4).count()# consumo de alcohol entre semana
    atypic_Dalc_5=df.filter(df['Dalc'] == 5).count()
    atypic_absences=df.filter(df['absences'] > 16).count()#numero de ausencias escolares

    print("atypic_age_22:",atypic_age_22)
    print("atypic_p_status_0:",atypic_p_status_0)
    print("atypic_travel_time_4:",atypic_travel_time_4)
    print("atypic_studytime_4:",atypic_studytime_4)
    print("atypic_failures_1:",atypic_failures_1)
    print("atypic_failures_2:",atypic_failures_2)
    print("atypic_failures_3:",atypic_failures_3)
    print("atypic_schoolsup_1:",atypic_schoolsup_1)
    print("atypic_paid_1:",atypic_paid_1)
    print("atypic_nursery_0:",atypic_nursery_0)
    print("atypic_higher_0:",atypic_higher_0)
    print("atypic_internet_0:",atypic_internet_0)
    print("atypic_famrel_1:",atypic_famrel_1)
    print("atypic_famrel_2:",atypic_famrel_2)
    print("atypic_freetime_1:",atypic_freetime_1)
    print("atypic_Dalc_4:",atypic_Dalc_4)
    print("atypic_Dalc_5:",atypic_Dalc_5)
    print("atypic_absences:",atypic_absences)

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

#-------------------------REGRESION LOGISTICA-----------------------------#
#NOTA: En este modelo se intento variar el parametro de maxIter en ambos datasets,
#pero este parametro no afectaba el desempeño del modelo en este caso.

def logistic_Regression(df,trainingData,testData,maxIterValue,thresholdValue,familyValue):

    print("\n")
    print("logistic_Regression")

    lr = LogisticRegression(labelCol="G3", featuresCol="features",maxIter=maxIterValue,
                                            threshold=thresholdValue, family=familyValue)

   # Fit the model

    model = lr.fit(trainingData)

   # make predictions using our trained model

    predictions = model.transform(testData)

    # estimate the accuracy of the prediction

    multi_evaluator = MulticlassClassificationEvaluator(labelCol="G3", predictionCol="prediction", metricName="accuracy")
    accuracy = multi_evaluator.evaluate(predictions)

    multi_evaluator = multi_evaluator.setMetricName('precisionByLabel')
    precision = multi_evaluator.evaluate(predictions)

    multi_evaluator = multi_evaluator.setMetricName('recallByLabel')
    recall = multi_evaluator.evaluate(predictions)
    
    multi_evaluator = multi_evaluator.setMetricName('f1')
    f1_score = multi_evaluator.evaluate(predictions)

    bin_evaluator = BinaryClassificationEvaluator(labelCol="G3", rawPredictionCol="prediction", metricName="areaUnderROC")
    area = bin_evaluator.evaluate(predictions)


    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1 score = {}".format(f1_score))
    print("Area under ROC curve = {}".format(area))

    return (model)

#----------------------------MACHINE LEARNING--------------------------------#

#-----------------------------RANDOM FOREST----------------------------------#

# Random Forest

def random_Forest(df,trainingData,testData,numTreesValue,maxDepthValue,featureSubsetStrategyValue):
    
    print("\n")
    print("random_Forest")
    rf = RandomForestClassifier(labelCol="G3", featuresCol="features", numTrees=numTreesValue,maxDepth=maxDepthValue,
                                             featureSubsetStrategy=featureSubsetStrategyValue)

    # Fit the model
    model = rf.fit(trainingData)

    # make predictions using our trained model
    predictions = model.transform(testData)

    # estimate the accuracy of the prediction
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="G3", predictionCol="prediction", metricName="accuracy")
    accuracy = multi_evaluator.evaluate(predictions)

    multi_evaluator = multi_evaluator.setMetricName('precisionByLabel')
    precision = multi_evaluator.evaluate(predictions)
        
    multi_evaluator = multi_evaluator.setMetricName('recallByLabel')
    recall = multi_evaluator.evaluate(predictions)

    multi_evaluator = multi_evaluator.setMetricName('f1')
    f1_score = multi_evaluator.evaluate(predictions)
    bin_evaluator = BinaryClassificationEvaluator(labelCol="G3", rawPredictionCol="prediction", metricName="areaUnderROC")
    area = bin_evaluator.evaluate(predictions)
    
    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1 score = {}".format(f1_score))
    print("Area under ROC curve = {}".format(area))
    
    # print model summary
    return (model)


#--------------------VECTOR DE MÁQUINA DE SOPORTES------------------------#
#NOTA: Este modelo cuenta con los siguientes párametros predeterminados:
#      maxIter = 100
#      threshold = 0.0
#      aggregationDepth = 0.2
#      regParam = 0.0
#------------------------------Dataset 1----------------------------------#

def svm(df,trainingData,testData,maxIterValue,regParamValue,depth,thresholdValue):

    print("\n")
    print("mvs")
            
    svm = LinearSVC(labelCol="G3", featuresCol="features", maxIter=maxIterValue, regParam=regParamValue, aggregationDepth=depth, threshold=thresholdValue)

    # Fit the model
    model = svm.fit(trainingData)

    # make predictions using our trained model

    predictions = model.transform(testData)

    # look at the result
    predictions.select("prediction", "G3").show(5)

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
    df=spark.read.csv('student-por.csv',sep=';',header=True)
    
    """
    #Visualización del dataframe
    df.show()

    #Tamaño del dataset
    print(df.count())
    El dataframe tiene 649 registros

    #Tipo de dato de cada variable
    print(df.dtypes)
    #NOTA:Todos los datos del dataframe inicial son de tipo string

    """

    #Cantidad de datos nulos 
    nullData(df)
    #Reemplazar valores categoricos a numericos
    df=categoricalToNumerical(df)

    #Convertir los datos de string a int
    df=stringToInt(df)

    #Convertir variables categorica a numericas
    df=approvedOrReproved(df)
    
    #Visualizar la correlación de las variables
    #correlacion = correlation(df,headings)
    #sns.heatmap(correlacion, square=True)

    #describeData(df)
    #boxWhiskerPlot(df,headings)

    countAtypicValues(df)

    df=dropAtypicValues(df)

    df=dataBalancing(df)

    #------------------------CREACIÓN DE LOS DATASETS FINALES---------------------#

    #Crear vectores assembler
    vector1 = VectorAssembler(inputCols=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob',
        'Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities',
        'nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences'],
        outputCol="features")

    #Adaptar los vectores al conjunto de datos

    df1_temp = vector1.transform(df)
    
    #df_temp.show(5)
    # get dataframe with all necessary data in the appropriate form

    df1 = df1_temp.drop('school','sex','age','address','famsize','Pstatus','Medu','Fedu',
        'Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup',
        'famsup','paid','activities','nursery','higher','internet','romantic',
        'famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2')
        
    #Partición de los dataframes

    trainingData_1, testData_1= df1.randomSplit([0.7,0.3],seed=2102020)

    print("Sin G1 y G2")

    #rl1_model1=logistic_Regression(df1,trainingData_1,testData_1,maxIterValue=50,thresholdValue=0.5,familyValue='binomial')
    rl1_model1=logistic_Regression(df1,trainingData_1,testData_1,maxIterValue=50,thresholdValue=0.55,familyValue='binomial')
    rl2_model1=logistic_Regression(df1,trainingData_1,testData_1,maxIterValue=100,thresholdValue=0.5,familyValue='binomial')
    rl3_model1=logistic_Regression(df1,trainingData_1,testData_1,maxIterValue=100,thresholdValue=0.55,familyValue='binomial')
    rl4_model1=logistic_Regression(df1,trainingData_1,testData_1,maxIterValue=150,thresholdValue=0.55,familyValue='binomial')
    #rl3_model1=logistic_Regression(df1,trainingData_1,testData_1,maxIterValue=50,thresholdValue=0.4,familyValue='binomial')

    print("\n")

    print("Sin G1 y G2")

    rf1_model1=random_Forest(df1,trainingData_1,testData_1, numTreesValue=10,maxDepthValue=5,
                                            featureSubsetStrategyValue='sqrt')
    rf2_model1=random_Forest(df1,trainingData_1,testData_1, numTreesValue=10,maxDepthValue=5,
                                            featureSubsetStrategyValue='onethird')
    rf3_model1=random_Forest(df1,trainingData_1,testData_1, numTreesValue=10,maxDepthValue=5,
                                            featureSubsetStrategyValue='log2')
    rf4_model1=random_Forest(df1,trainingData_1,testData_1, numTreesValue=10,maxDepthValue=3,
                                            featureSubsetStrategyValue='sqrt')
    rf4_model1=random_Forest(df1,trainingData_1,testData_1, numTreesValue=10,maxDepthValue=8,
                                            featureSubsetStrategyValue='sqrt')

    print("Sin G1 y G2")
    mvs1_model1=svm(df1,trainingData_1,testData_1,maxIterValue=100,thresholdValue = 0.0, depth = 2, regParamValue=0)
    mvs2_model1=svm(df1,trainingData_1,testData_1,maxIterValue=10,thresholdValue = 0.0, depth = 2, regParamValue=0)
    mvs3_model1=svm(df1,trainingData_1,testData_1,maxIterValue  = 100, thresholdValue = 0.0, depth = 2, regParamValue=0.1)
    mvs4_model1=svm(df1,trainingData_1,testData_1, maxIterValue =10, thresholdValue = 0.0, depth = 2,  regParamValue=0.1)
    mvs5_model1=svm(df1,trainingData_1,testData_1, maxIterValue =10, thresholdValue = 0.0, depth = 2, regParamValue = 0.0)
    mvs6_model1=svm(df1,trainingData_1,testData_1, maxIterValue =150, thresholdValue = 0.0, depth = 2, regParamValue = 0.0)
    mvs7_model1=svm(df1,trainingData_1,testData_1, maxIterValue  = 100, thresholdValue=0.5, depth = 2, regParamValue = 0.0)
    mvs8_model1=svm(df1,trainingData_1,testData_1, maxIterValue =10, thresholdValue=0.5, depth = 2, regParamValue = 0.0)
    mvs9_model1=svm(df1,trainingData_1,testData_1, maxIterValue  = 100, thresholdValue = 0.0, depth=3, regParamValue = 0.0)
  

    print("\n")

    print("#------------------------------------------------------------------------------------------#")

    #Crear vectores assembler

    vector2 = VectorAssembler(inputCols=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob',
        'Fjob','reason','guardian','traveltime','studytime','failures','schoolsup',
        'famsup','paid','activities','nursery','higher','internet','romantic',
        'famrel','freetime','goout','Dalc','Walc','health','absences','G1'], outputCol="features")

    #Adaptar los vectores al conjunto de datos

    df2_temp = vector2.transform(df)

    #df2_temp.show(5)
    # get dataframe with all necedf2_tempssary data in the appropriate form

    df2 = df2_temp.drop('school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob',
        'Fjob','reason','guardian','traveltime','studytime','failures','schoolsup',
        'famsup','paid','activities','nursery','higher','internet','romantic',
        'famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2')

    #Partición de los dataframes
    trainingData_2, testData_2= df1.randomSplit([0.7,0.3],seed=2102020)

    print("Sin G2")
    rl_model2=logistic_Regression(df2,trainingData_2,testData_2,maxIterValue=50,thresholdValue=0.5,familyValue='binomial')
    print("\n")

    print("Sin G2")
    rf_model2=random_Forest(df2,trainingData_2,testData_2, numTreesValue=10,maxDepthValue=8,
                                         featureSubsetStrategyValue='log2')
    print("\n")

    print("Sin G2")
    mvs_model2=svm(df2,trainingData_2,testData_2,maxIterValue=50,regParamValue=0.1)
    print("\n")

    #Finaliza la sesión de spark
    spark.stop()

main()