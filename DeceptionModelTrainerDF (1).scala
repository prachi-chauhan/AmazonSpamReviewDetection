// Databricks notebook source
import sys.process._

import org.apache.spark.sql.functions.concat_ws
import org.apache.spark.ml.classification.{ MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel, NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}

import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import java.util.Properties

// COMMAND ----------

// Download deception data move it to FileStore

dbutils.fs.mkdirs("/FileStore/deception_dataset")

var result = dbutils.fs.ls("/FileStore/deception_dataset").toString

// checking id both types of data exist or not
if (!result.contains("dataset/d") || !result.contains("dataset/t")) {
  result = (("ls /tmp") !).toString
  
  //println(result)
  
  dbutils.fs.rm("/FileStore/deception_dataset", true)
  
  dbutils.fs.mkdirs("/FileStore/deception_dataset")
  
  // if zip is still in tmp
  if (result.contains("deception.zip") ) {
    println("Unziping deception.zip ...")
    
    "unzip -o /tmp/deception.zip -d /tmp" !!
    
    println("Copying files to FileStore ...")
    
    dbutils.fs.cp("file:/tmp/deception", "/FileStore/deception_dataset", true)
    
    "rm -rf /tmp/deception" !!
    
    println("Moved files to FileStore")
  }
  else {
    // If zip not found then download zip file
    println("Downloading deception.zip ...")
    
    result = (("wget -q -P /tmp https://raw.githubusercontent.com/kartheek2002/BigData/master/deception.zip") !).toString
    
    if(result == "0") {
      println("Unziping deception.zip ...")
      
      "unzip -o /tmp/deception.zip -d /tmp" !!
      
      println("Copying files to FileStore ...")
      
      dbutils.fs.cp("file:/tmp/deception", "/FileStore/deception_dataset", true)
      
      "rm -rf /tmp/deception" !!
      
      println("Downloaded and moved file to FileStore")
    }
    else {
      println("Downloaded failed")
      System.exit(1)
    }
  }
}
else {
  var tCount = dbutils.fs.ls("/FileStore/deception_dataset/t").length
  var dCount = dbutils.fs.ls("/FileStore/deception_dataset/d").length
  
  
  // Checking if all files are there. If not reupload files.
  if(tCount == 800 && dCount == 1098) {
    println("File already in FileStore")
  }
  else {
    dbutils.fs.rm("/FileStore/deception_dataset", true)
  
    dbutils.fs.mkdirs("/FileStore/deception_dataset")

    if (result.contains("deception.zip") ) {
      println("Unziping deception.zip ...")
      
      "unzip -o /tmp/deception.zip -d /tmp" !!
      
      println("Copying files to FileStore ...")
      
      dbutils.fs.cp("file:/tmp/deception", "/FileStore/deception_dataset", true)

      "rm -rf /tmp/deception" !!

      println("Moved files to FileStore")
    }
    else {
      println("Downloading deception.zip ...")

      result = (("wget -q -P /tmp https://raw.githubusercontent.com/kartheek2002/BigData/master/deception.zip") !).toString

      if(result == "0") {
        println("Unziping deception.zip ...")
        
        "unzip -o /tmp/deception.zip -d /tmp" !!
        
        println("Copying files to FileStore ...")
        
        dbutils.fs.cp("file:/tmp/deception", "/FileStore/deception_dataset", true)

        "rm -rf /tmp/deception" !!

        println("Downloaded and moved file to FileStore")
      }
      else {
        println("Downloaded failed")
        System.exit(1)
      }
    }
  }
}

// COMMAND ----------

// Load deception data
val deceptionRDD = sc.textFile("/FileStore/deception_dataset/d").zipWithIndex().map(r => ("D" + r._2.toString, r._1.toString, 0))
val truthfulRDD = sc.textFile("/FileStore/deception_dataset/t").zipWithIndex().map(r => ("T" + r._2.toString, r._1.toString, 1))

// COMMAND ----------

// join both data
val allRDD = deceptionRDD.union(truthfulRDD)

// COMMAND ----------

// Stanford CoreNLP method to remove stop words and normalize (lemma) a line
val stopWords = Set("stopWord")

def plainTextToLemmas(text: String, stopWords: Set[String]): Seq[String] = {
  val props = new Properties()
  props.put("annotators", "tokenize, ssplit, pos, lemma")
  val pipeline = new StanfordCoreNLP(props)
  val doc = new Annotation(text.replaceAll("[,.!?:;]", ""))
  pipeline.annotate(doc)
  val lemmas = new ArrayBuffer[String]()

  val sentences = doc.get(classOf[SentencesAnnotation])
  for (sentence <- sentences; token <- sentence.get(classOf[TokensAnnotation])) {
    val lemma = token.get(classOf[LemmaAnnotation])
    if (lemma.length > 2 && !stopWords.contains(lemma)) {
      lemmas += lemma.toLowerCase
    }
  }
  lemmas
}

val lemmatized = allRDD.map(r => (r._1, plainTextToLemmas(r._2.toString, stopWords).map(_.mkString("")).mkString(" "), r._3))

// COMMAND ----------

val lemmatizedDF = lemmatized.toDF("id", "value", "label")
//lemmatizedDF.take(1).foreach(println)

// COMMAND ----------

val tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words")
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(5000)
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

val wordsData = tokenizer.transform(lemmatizedDF)
val featurizedData = hashingTF.transform(wordsData)
val idfModel = idf.fit(featurizedData)

val rescaledData = idfModel.transform(featurizedData).select("label", "features")
rescaledData.printSchema()

// COMMAND ----------

// 60-40 split of data
val Array(training, test) = rescaledData.randomSplit(Array(0.6, 0.4))

// COMMAND ----------

val layers = Array[Int](5000, 2)

val mpc = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(12345L)
  .setMaxIter(400)

val pipeline = new Pipeline().setStages(Array(mpc))

// Fit the pipeline to training documents.
val model = pipeline.fit(training)

// COMMAND ----------

val predictions = model.transform(test)
predictions.printSchema()

// COMMAND ----------

val evaluator1 = new MulticlassClassificationEvaluator().setMetricName("accuracy");
val evaluator2 = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision");
val evaluator3 = new MulticlassClassificationEvaluator().setMetricName("weightedRecall");
val evaluator4 = new MulticlassClassificationEvaluator().setMetricName("f1");

println("-------------------------------------------------------------------")
println("Deception Model Metrics:")
println("-------------------------------------------------------------------")
println("Accuracy = " + evaluator1.evaluate(predictions))
println("Precision = " + evaluator2.evaluate(predictions))
println("Recall = " + evaluator3.evaluate(predictions))
println("F1 = " + evaluator4.evaluate(predictions))
println("-------------------------------------------------------------------")

// COMMAND ----------

// Store model to file to be used with Amazon review labeling
model.write.overwrite.save("/FileStore/deception_dataset/model/MultilayerPerceptronModel_withoutSymbolsStopWordsLemmaDF")
