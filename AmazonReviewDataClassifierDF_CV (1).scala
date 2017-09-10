// Databricks notebook source
import edu.stanford.nlp.ling.CoreAnnotations._
import edu.stanford.nlp.pipeline._

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Properties
import java.util.concurrent.TimeUnit
import java.util.Calendar

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LinearSVC, MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel, RandomForestClassifier, NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, NGram, Tokenizer, StringIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.Column
import org.apache.spark.sql.functions.{coalesce, col, monotonically_increasing_id}

import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer

import scala.util.{Success, Try}

// COMMAND ----------

// Count words in given string
def countWords(r: String) : Int = {
  return r.split(" ").size.toInt
}

// Get capital letter ratio (treats symbols as capital letters)
def getCapitalRatio(r: String): Double = {
  val letters = r.replaceAll(" ", "").split("")
  
  var upper = 0
  
  for( letter <- letters) {
    if(letter == letter.toUpperCase()) {
      upper = upper + 1
    }
  }
  
  return BigDecimal(upper.toDouble/letters.length).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
}

// Get number of days using the list of UnixReivewTime
def getNumDays(r: List[Integer]): Int = {
  val df = new SimpleDateFormat("yyyy-MM-dd")

  val ts1 = r(0) * 1000L
  val date1 = df.format(ts1)
  //println(date1)

  val ts2 = r.last * 1000L
  val date2 = df.format(ts2)
  //println(date2)

  val d1:Date = df.parse(df.format(ts1))
  val d2:Date = df.parse(df.format(ts2))
  val diff:Long = d2.getTime() - d1.getTime();
  return ((TimeUnit.DAYS.convert(diff, TimeUnit.MILLISECONDS)) + 1).toInt
}

// Get product star info
def productjoin(rdd: RDD[(Integer,String,Integer,Integer,Double,String,String,String,String,String,Integer,Double)]):RDD[(String, (Double, Double, Double, Double, Double))]={
 
  val r1 = rdd.filter(x=>(x._5.toString.toDouble == 1)).map(x=>(x._2,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  val r2 = rdd.filter(x=>(x._5.toString.toDouble == 2)).map(x=>(x._2,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  val r3 = rdd.filter(x=>(x._5.toString.toDouble == 3)).map(x=>(x._2,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  val r4 = rdd.filter(x=>(x._5.toString.toDouble == 4)).map(x=>(x._2,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  val r5 = rdd.filter(x=>(x._5.toString.toDouble == 5)).map(x=>(x._2,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  val cr = rdd.map(x=>(x._2,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
 
  val r1DF = r1.toDF("ID1", "1star")
  val r2DF = r2.toDF("ID2", "2star")
  val r3DF = r3.toDF("ID3", "3star")
  val r4DF = r4.toDF("ID4", "4star")
  val r5DF = r5.toDF("ID5", "5star")
  val crDF = cr.toDF("ID6", "total")

  val join = r1DF.join(r2DF, $"ID1" === $"ID2", "fullouter")
              .select(coalesce($"ID1", $"ID2").alias("ID"), $"1star", $"2star")
              .join(r3DF, $"ID" === $"ID3", "fullouter")
              .select(coalesce($"ID", $"ID3").alias("ID"), $"1star", $"2star", $"3star")
              .join(r4DF, $"ID" === $"ID4", "fullouter")
              .select(coalesce($"ID", $"ID4").alias("ID"), $"1star", $"2star", $"3star",$"4star")
              .join(r5DF, $"ID" === $"ID5", "fullouter")
              .select(coalesce($"ID", $"ID5").alias("ID"), $"1star", $"2star", $"3star",$"4star",$"5star")
              .join(crDF, $"ID" === $"ID6", "fullouter").na.fill(0)
              .select(coalesce($"ID", $"ID6").alias("ID"), $"1star", $"2star", $"3star",$"4star",$"5star",$"total")
              .as[(String, Int, Int, Int, Int, Int, Int)]
              .rdd
              .map(x => (x._1, ((Math.pow(x._2, 2) / x._7 * 20),(Math.pow(x._3, 2) / x._7 * 40),(Math.pow(x._4, 2) / x._7 * 60),
                                (Math.pow(x._5, 2) / x._7 * 80),(Math.pow(x._6, 2) / x._7 * 100))))
  
  return join
}

// Get reviwer star info
def userjoin(rdd: RDD[(Integer,String,Integer,Integer,Double,String,String,String,String,String,Integer,Double)]):RDD[(String, (Double, Double, Double, Double, Double))]={
 
  val r1 = rdd.filter(x=>(x._5.toString.toDouble == 1)).map(x=>(x._8,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  val r2 = rdd.filter(x=>(x._5.toString.toDouble == 2)).map(x=>(x._8,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  val r3 = rdd.filter(x=>(x._5.toString.toDouble == 3)).map(x=>(x._8,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  val r4 = rdd.filter(x=>(x._5.toString.toDouble == 4)).map(x=>(x._8,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  val r5 = rdd.filter(x=>(x._5.toString.toDouble == 5)).map(x=>(x._8,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  val cr = rdd.map(x=>(x._8,1)).reduceByKey(_+_).groupByKey.map(r => (r._1, r._2.last)).sortByKey()
  
  val r1DF = r1.toDF("ID1", "1star")
  val r2DF = r2.toDF("ID2", "2star")
  val r3DF = r3.toDF("ID3", "3star")
  val r4DF = r4.toDF("ID4", "4star")
  val r5DF = r5.toDF("ID5", "5star")
  val crDF = cr.toDF("ID6", "total")

  val join = r1DF.join(r2DF, $"ID1" === $"ID2", "fullouter")
                .select(coalesce($"ID1", $"ID2").alias("ID"), $"1star", $"2star")
                .join(r3DF, $"ID" === $"ID3", "fullouter")
                .select(coalesce($"ID", $"ID3").alias("ID"), $"1star", $"2star", $"3star")
                .join(r4DF, $"ID" === $"ID4", "fullouter")
                .select(coalesce($"ID", $"ID4").alias("ID"), $"1star", $"2star", $"3star",$"4star")
                .join(r5DF, $"ID" === $"ID5", "fullouter")
                .select(coalesce($"ID", $"ID5").alias("ID"), $"1star", $"2star", $"3star",$"4star",$"5star")
                .join(crDF, $"ID" === $"ID6", "fullouter").na.fill(0)
                .select(coalesce($"ID", $"ID6").alias("ID"), $"1star", $"2star", $"3star",$"4star",$"5star",$"total")
                .as[(String, Int, Int, Int, Int, Int, Int)]
                .rdd
                .map(x => (x._1, ((Math.pow(x._2, 2) / x._7 * 20),(Math.pow(x._3, 2) / x._7 * 40),(Math.pow(x._4, 2) / x._7 * 60),
                                  (Math.pow(x._5, 2) / x._7 * 80),(Math.pow(x._6, 2) / x._7 * 100))))
  
  return join
}

// Stanford CoreNLP method to remove stopwords and normalize (lemma)
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

// COMMAND ----------

// File name variables
var fileLink = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz"
var fileName = "reviews_Digital_Music_5.json.gz"
var tempFolder = "/tmp"
var fileStoreFolder = "/FileStore/reviews_Digital_Music_5"

// COMMAND ----------

val reviewsLabeledDF = sqlContext.read.format("com.databricks.spark.csv")
                  .option("header", "true")
                  .option("inferSchema", "true")
                  .load("/FileStore/LabeledData/" + fileStoreFolder.split("/")(2) + "/*.csv")
reviewsLabeledDF.printSchema()

// COMMAND ----------

// DF to RDD
val reviewsLabeledRDD = reviewsLabeledDF.as[(Integer,String,Integer,Integer,Double,String,String,String,String,String,Integer,Double)].rdd
//reviewsLabeledRDD.take(1).foreach(println)

// COMMAND ----------

// get user and product start info
val productStarInfoDF = productjoin(reviewsLabeledRDD)
                          .map(r => (r._1, r._2._1, r._2._2, r._2._3, r._2._4, r._2._5))
                          .toDF("pid", "p1star", "p2star", "p3star", "p4star", "p5star")

val userStarInfoDF = userjoin(reviewsLabeledRDD)
                        .map(r => (r._1, r._2._1, r._2._2, r._2._3, r._2._4, r._2._5))
                        .toDF("uid", "u1star", "u2star", "u3star", "u4star", "u5star")

productStarInfoDF.printSchema()
userStarInfoDF.printSchema()

//productStarInfoDF.take(1).foreach(println)
//userStarInfoDF.take(1).foreach(println)

// COMMAND ----------

// join reviews with product and user star info
//println(reviewsLabeledDF.count)

val reviewsProductUserLabeledDF = (reviewsLabeledDF.join(productStarInfoDF, $"asin" === $"pid", "inner"))
                                    .join(userStarInfoDF, $"reviewerID" === $"uid", "inner")
                                    .select("reviewID", "asin", "helpfulPos", "helpfulNeg", "overall", "reviewText", "unixReviewTime", "reviewerID", 
                                            "p1star", "p2star", "p3star", "p4star", "p5star", 
                                            "u1star", "u2star", "u3star", "u4star", "u5star", 
                                            "label")

reviewsProductUserLabeledDF.printSchema()
//println(reviewsProductUserLabeledDF.count())
//reviewsProductUserLabeledDF.take(1).foreach(println)

// COMMAND ----------

// DF to RDD
val reviewsProductUserLabeledRDD = reviewsProductUserLabeledDF
                                    .as[(Integer,String,Integer,Integer,Double,String,Integer,String,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double)]
                                    .rdd
                                    .map(r => (r._1, r._2, (r._3 - r._4), r._5, r._6, r._7, r._8, r._9, r._10, r._11, r._12, r._13, r._14, r._15, r._16, r._17, r._18, r._19))

//reviewsProductUserLabeledRDD.take(1).foreach(println)

// COMMAND ----------

// get user review rate per day
val userTotalReviews = reviewsProductUserLabeledRDD
                        .map(r => (r._7,  1))
                        .reduceByKey(_ + _)

val userStartEndDate = reviewsProductUserLabeledRDD
                          .map(r => (r._7, r._6))
                          .groupByKey()
                          .map(r => (r._1, getNumDays(r._2.toList.sorted)))

val userReviewRate = userTotalReviews
                        .join(userStartEndDate)
                         .map(r => (r._1, r._2._1.toDouble/r._2._2.toDouble))

/*
println("------------------------------------")
userTotalReviews.take(5).foreach(println)
println("------------------------------------")
userStartEndDate.take(5).foreach(println)
println("------------------------------------")
userReviewRate.take(5).foreach(println)
println("------------------------------------")
*/

// COMMAND ----------

val userReviewRateDF = userReviewRate.toDF("reviewerID", "reviewRate")
//userReviewRateDF.orderBy(col("reviewRate").desc).show()

// COMMAND ----------

// Order of features: reviewID, helpfulness, rating, capitalLetterRatio, reviewWordCount, userReviewRate, pstar1, pstar2, pstar3, pstar4, pstar5, ustar1, ustar2, ustar3, ustar4, ustar5, label
val reviewsFeaturesRDD = reviewsProductUserLabeledRDD
                            .map(r => (r._7, (r._1, r._3, r._4, getCapitalRatio(r._5), countWords(r._5), 
                                       r._8, r._9, r._10, r._11, r._12, 
                                       r._13, r._14, r._15, r._16, r._17,
                                       r._18)))
                            .join(userReviewRate)
                            .map(r => (r._2._1._1.toDouble, (r._2._1._2.toDouble, r._2._1._3, r._2._1._4, r._2._1._5.toDouble, 
                                       r._2._1._6, r._2._1._7, r._2._1._8, r._2._1._9, r._2._1._10, 
                                       r._2._1._11, r._2._1._12, r._2._1._13, r._2._1._14, r._2._1._15, 
                                       r._2._2, r._2._1._16)))

//reviewsFeaturesRDD.take(1).foreach(println)

val reviewsFeaturesDF = reviewsFeaturesRDD
                          .map(r => (r._1, r._2._1, r._2._2, r._2._3, r._2._4, 
                                       r._2._5, r._2._6, r._2._7, r._2._8, r._2._9, 
                                       r._2._10, r._2._11, r._2._12, r._2._13, r._2._14, 
                                       r._2._15, r._2._16))
                          .toDF("reviewID","helpfulness","rating","capitalLetterRatio","reviewWordCount",
                                "pstar1","pstar2","pstar3","pstar4","pstar5",
                                "ustar1","ustar2","ustar3","ustar4","ustar5",
                                "userReviewRate", "label")

reviewsFeaturesDF.printSchema()
//reviewsFeaturesDF.take(1).foreach(println)

// COMMAND ----------

val reviewTextNLPRDD = reviewsLabeledDF.select($"reviewID", $"reviewText").as[(Double, String)].rdd.map(r => (r._1, plainTextToLemmas(r._2.toString, stopWords).map(_.mkString("")).mkString(" ")))

//reviewTextNLPRDD.take(1).foreach(println)

val textFeatJoinRDD = reviewTextNLPRDD.join(reviewsFeaturesRDD)

val textOnlyRDD = textFeatJoinRDD.map(r => (r._2._2._16, r._2._1))

val textAndFeaturesRDD = textFeatJoinRDD
                            .map(r => (r._2._2._16, (r._2._2._1.toString + " " + r._2._2._2.toString + " " + r._2._2._3.toString + " " + r._2._2._4.toString + " " + 
                                       r._2._2._5.toString + " " + r._2._2._6.toString + " " + r._2._2._7.toString + " " + r._2._2._8.toString + " " + r._2._2._9.toString + " " + 
                                       r._2._2._10.toString + " " + r._2._2._11.toString + " " + r._2._2._12.toString + " " + r._2._2._13.toString + " " + r._2._2._14.toString + " " + 
                                       r._2._2._5.toString + " " + 
                                       r._2._1)))

val textOnlyDF = textOnlyRDD.toDF("label", "text")
val textAndFeaturesDF = textAndFeaturesRDD.toDF("label", "text")

// COMMAND ----------

//textOnlyDF.take(1).foreach(println)
//textAndFeaturesDF.take(1).foreach(println)

// COMMAND ----------

val tokenizer = new Tokenizer()
  .setInputCol("text")
  .setOutputCol("words")

val hashingTF = new HashingTF()
  .setNumFeatures(5000)
  .setInputCol(tokenizer.getOutputCol)
  .setOutputCol("rawFeatures")

val hashingIDF = new IDF()
  .setInputCol("rawFeatures")
  .setOutputCol("features")

// COMMAND ----------

val tokenData1 = tokenizer.transform(textOnlyDF)
val tokenData2 = tokenizer.transform(textAndFeaturesDF)

val featurizedData1 = hashingTF.transform(tokenData1)
val featurizedData2 = hashingTF.transform(tokenData2)

val idfModel1 = hashingIDF.fit(featurizedData1)
val idfModel2 = hashingIDF.fit(featurizedData2)

val textOnlyHashedDF = idfModel1.transform(featurizedData1).select("label", "features").cache()
val textAndFeaturesHashedDF = idfModel2.transform(featurizedData2).select("label", "features").cache()

// COMMAND ----------

val Array(t1, t2) = textOnlyHashedDF.randomSplit(Array(0.6, 0.4))
val Array(tf1, tf2) = textAndFeaturesHashedDF.randomSplit(Array(0.6, 0.4))

val textOnlyTraining = t1.cache()
val textOnlyTest = t2.cache()
val textFeatTraining = tf1.cache()
val textFeatTest = tf2.cache()

// COMMAND ----------

// Select (prediction, true label) and compute test error
val evaluator1 = new MulticlassClassificationEvaluator().setMetricName("accuracy");
val evaluator2 = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision");
val evaluator3 = new MulticlassClassificationEvaluator().setMetricName("weightedRecall");
val evaluator4 = new MulticlassClassificationEvaluator().setMetricName("f1");

// COMMAND ----------

def nbClass() = {
  println("-------------------------------------------------------------------")
  println("Executing NaiveBayes Classifier...")
  println("-------------------------------------------------------------------")

  val nb = new NaiveBayes()
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setModelType("multinomial")

  val nbPipeline = new Pipeline()
    .setStages(Array(nb))

  // Fit the pipeline to training documents.
  val nbmodel1 = nbPipeline.fit(textOnlyTraining)
  val nbmodel2 = nbPipeline.fit(textFeatTraining)

  val nbPred1 = nbmodel1.transform(textOnlyTest)
  val nbPred2 = nbmodel2.transform(textFeatTest)

  println("Text Only Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(nbPred1))
  println("Precision = " + evaluator2.evaluate(nbPred1))
  println("Recall = " + evaluator3.evaluate(nbPred1))
  println("F1 = " + evaluator4.evaluate(nbPred1))
  
  println("-------------------------------------------------------------------")
  println("Text & Features Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(nbPred2))
  println("Precision = " + evaluator2.evaluate(nbPred2))
  println("Recall = " + evaluator3.evaluate(nbPred2))
  println("F1 = " + evaluator4.evaluate(nbPred2))
  
  println("-------------------------------------------------------------------")
}

// COMMAND ----------

def kmClass() = {
  println("-------------------------------------------------------------------")
  println("Executing KMeans Classifier...")
  println("-------------------------------------------------------------------")
  
  // Trains a k-means model.
  val km = new KMeans().setK(2).setMaxIter(100)

  val kModel1 = km.fit(textOnlyTraining)
  val kModel2 = km.fit(textFeatTraining)

  // Evaluate clustering by computing Within Set Sum of Squared Errors.
  val WSSSE1 = kModel1.computeCost(textOnlyTraining)
  println(s"Text Only Dataset Within Set Sum of Squared Errors = $WSSSE1")

  val WSSSE2 = kModel2.computeCost(textFeatTraining)
  println(s"Text & Features Dataset Within Set Sum of Squared Errors = $WSSSE2")

  val kMeansPred1 = kModel1.transform(textOnlyTest).withColumn("prediction2", 'prediction.cast("Double")).select('label, 'prediction2 as 'prediction)
  val kMeansPred2 = kModel2.transform(textFeatTest).withColumn("prediction2", 'prediction.cast("Double")).select('label, 'prediction2 as 'prediction)

  println("-------------------------------------------------------------------")
  println("Text Only Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(kMeansPred1))
  println("Precision = " + evaluator2.evaluate(kMeansPred1))
  println("Recall = " + evaluator3.evaluate(kMeansPred1))
  println("F1 = " + evaluator4.evaluate(kMeansPred1))
  
  println("-------------------------------------------------------------------")
  println("Text & Features Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(kMeansPred2))
  println("Precision = " + evaluator2.evaluate(kMeansPred2))
  println("Recall = " + evaluator3.evaluate(kMeansPred2))
  println("F1 = " + evaluator4.evaluate(kMeansPred2))
  
  println("-------------------------------------------------------------------")
}

// COMMAND ----------

def rfClass() = {
  println("-------------------------------------------------------------------")
  println("Executing RandomForest Classifier...")
  println("-------------------------------------------------------------------")
  
  val rf = new RandomForestClassifier()
                  .setImpurity("gini")
                  .setNumTrees(10)
                  .setMaxDepth(10)
                  .setMaxBins(32)

  val rfPipeline = new Pipeline().setStages(Array(rf))

  // Fit the pipeline to training documents.
  val rfModel1 = rfPipeline.fit(textOnlyTraining)
  val rfModel2 = rfPipeline.fit(textFeatTraining)

  val rfPred1 = rfModel1.transform(textOnlyTest)
  val rfPred2 = rfModel2.transform(textFeatTest)

  println("Text Only Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(rfPred1))
  println("Precision = " + evaluator2.evaluate(rfPred1))
  println("Recall = " + evaluator3.evaluate(rfPred1))
  println("F1 = " + evaluator4.evaluate(rfPred1))
  
  println("-------------------------------------------------------------------")
  println("Text & Features Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(rfPred2))
  println("Precision = " + evaluator2.evaluate(rfPred2))
  println("Recall = " + evaluator3.evaluate(rfPred2))
  println("F1 = " + evaluator4.evaluate(rfPred2))
  
  println("-------------------------------------------------------------------")
}

// COMMAND ----------

def lsvcClass() = {
  println("-------------------------------------------------------------------")
  println("Executing LinearSVC Classifier...")
  println("-------------------------------------------------------------------")
  
  val lsvc = new LinearSVC()
    .setMaxIter(10)
    .setRegParam(0.5)

  val lsvcPipeline = new Pipeline().setStages(Array(lsvc))

  // Fit the model
  val lsvcModel1 = lsvcPipeline.fit(textOnlyTraining)
  val lsvcModel2 = lsvcPipeline.fit(textFeatTraining)

  val lsvcPred1 = lsvcModel1.transform(textOnlyTest)
  val lsvcPred2 = lsvcModel2.transform(textFeatTest)

  println("Text Only Dataset Metrics:")    
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(lsvcPred1))
  println("Precision = " + evaluator2.evaluate(lsvcPred1))
  println("Recall = " + evaluator3.evaluate(lsvcPred1))
  println("F1 = " + evaluator4.evaluate(lsvcPred1))
  
  println("-------------------------------------------------------------------")
  println("Text & Features Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(lsvcPred2))
  println("Precision = " + evaluator2.evaluate(lsvcPred2))
  println("Recall = " + evaluator3.evaluate(lsvcPred2))
  println("F1 = " + evaluator4.evaluate(lsvcPred2))
  
  println("-------------------------------------------------------------------")
}

// COMMAND ----------

def mpcClass() = {
  println("-------------------------------------------------------------------")
  println("Executing MultilayerPerceptronClassifier Classifier...")
  println("-------------------------------------------------------------------")
  
  // specify layers for the neural network:
  // input layer of size 5000 (features)
  // and output of size 2 (classes)
  val layers = Array[Int](5000, 2)

  // create the trainer and set its parameters
  val mpc = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setBlockSize(128)
    .setSeed(12345L)
    .setMaxIter(400)

  // train the model
  val mpcModel1 = mpc.fit(textOnlyTraining)
  val mpcModel2 = mpc.fit(textFeatTraining)

  // compute accuracy on the test set
  val mpcPred1 = mpcModel1.transform(textOnlyTest)
  val mpcPred2 = mpcModel2.transform(textFeatTest)
  
  println("Text Only Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(mpcPred1))
  println("Precision = " + evaluator2.evaluate(mpcPred1))
  println("Recall = " + evaluator3.evaluate(mpcPred1))
  println("F1 = " + evaluator4.evaluate(mpcPred1))
  
  println("-------------------------------------------------------------------")
  println("Text & Features Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(mpcPred2))
  println("Precision = " + evaluator2.evaluate(mpcPred2))
  println("Recall = " + evaluator3.evaluate(mpcPred2))
  println("F1 = " + evaluator4.evaluate(mpcPred2))
  
  println("-------------------------------------------------------------------")
}

// COMMAND ----------

def nbClassCV() = {
  println("-------------------------------------------------------------------")
  println("Executing NaiveBayes Cross Validation Classifier...")
  println("-------------------------------------------------------------------")
  
  val nb = new NaiveBayes()
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setModelType("multinomial")
  
  val nFolds: Int = 7
  val paramGrid = new ParamGridBuilder()
   .addGrid(nb.smoothing,Array(0.0,0.5,1.0))
  .build()

  val nbPipeline = new Pipeline()
    .setStages(Array(nb))
  
  val cv = new CrossValidator()
  .setEstimator(nbPipeline)
  .setEvaluator(evaluator1) 
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(nFolds)

  val cvmodel1 = cv.fit(textOnlyTraining)
  val nbprediction1 = cvmodel1.transform(textOnlyTest)
  
  val cvmodel2 = cv.fit(textFeatTraining)
  val nbprediction2 = cvmodel2.transform(textFeatTest)

  println("Text Only Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(nbprediction1))
  println("Precision = " + evaluator2.evaluate(nbprediction1))
  println("Recall = " + evaluator3.evaluate(nbprediction1))
  println("F1 = " + evaluator4.evaluate(nbprediction1))
  
  println("-------------------------------------------------------------------")
  println("Text & Features Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(nbprediction2))
  println("Precision = " + evaluator2.evaluate(nbprediction2))
  println("Recall = " + evaluator3.evaluate(nbprediction2))
  println("F1 = " + evaluator4.evaluate(nbprediction2))
  
  println("-------------------------------------------------------------------")
}

// COMMAND ----------

def kmClassCV() = {
  println("-------------------------------------------------------------------")
  println("Executing KMeans Cross Validation Classifier...")
  println("-------------------------------------------------------------------")

  // Trains a k-means model.
  val km = new KMeans().setK(2).setMaxIter(100)

  val nFolds: Int = 7
  val metric: String = "accuracy"
  
  val labelIndexer = new StringIndexer().setInputCol("prediction").setOutputCol("prediction2")
  
  val pipeline = new Pipeline().setStages(Array(km, labelIndexer)) 
  val paramGrid = new ParamGridBuilder()
                .addGrid(km.maxIter,Array(100,400,500))
                .build()

  val evalKM = new MulticlassClassificationEvaluator().setPredictionCol("prediction2").setMetricName("accuracy");

  val cvkm = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evalKM) 
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(nFolds)

  val cvmodel1 = cvkm.fit(textOnlyTraining)
  val cvmodel2 = cvkm.fit(textFeatTraining)
  
  val kMeansPredCV1 = cvmodel1.transform(textOnlyTest).withColumn("prediction2", 'prediction.cast("Double")).select('label, 'prediction2 as 'prediction)
  val kMeansPredCV2  = cvmodel2.transform(textFeatTest).withColumn("prediction2", 'prediction.cast("Double")).select('label, 'prediction2 as 'prediction)

  println("Text Only Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(kMeansPredCV1))
  println("Precision = " + evaluator2.evaluate(kMeansPredCV1))
  println("Recall = " + evaluator3.evaluate(kMeansPredCV1))
  println("F1 = " + evaluator4.evaluate(kMeansPredCV1))

  println("-------------------------------------------------------------------")
  println("Text & Features Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(kMeansPredCV2))
  println("Precision = " + evaluator2.evaluate(kMeansPredCV2))
  println("Recall = " + evaluator3.evaluate(kMeansPredCV2))
  println("F1 = " + evaluator4.evaluate(kMeansPredCV2))
  
  println("-------------------------------------------------------------------")
}

// COMMAND ----------

def rfClassCV() = {
  println("-------------------------------------------------------------------")
  println("Executing RandomForest Cross Validation Classifier...")
  println("-------------------------------------------------------------------")
  
  val rf = new RandomForestClassifier().setImpurity("gini")

  val nFolds: Int = 7
  val paramGrid = new ParamGridBuilder()
  .addGrid(rf.maxDepth,Array(10,12,15))
  .addGrid(rf.numTrees,Array(10,12,15))
  .build()

  val rfPipeline = new Pipeline()
    .setStages(Array(rf))
  
  val rfcv = new CrossValidator()
  .setEstimator(rfPipeline)
  .setEvaluator(evaluator1) 
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(nFolds)
  
   // Fit the pipeline to training documents.
  val rfcvModel1 = rfcv.fit(textOnlyTraining)
  val rfcvModel2 = rfcv.fit(textFeatTraining)

  val rfcvPred1 = rfcvModel1.transform(textOnlyTest)
  val rfcvPred2 = rfcvModel2.transform(textFeatTest)

  println("Text Only Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(rfcvPred1))
  println("Precision = " + evaluator2.evaluate(rfcvPred1))
  println("Recall = " + evaluator3.evaluate(rfcvPred1))
  println("F1 = " + evaluator4.evaluate(rfcvPred1))
  
  println("-------------------------------------------------------------------")
  println("Text & Features Dataset Metrics:")
  println("-------------------------------------------------------------------")
  println("Accuracy = " + evaluator1.evaluate(rfcvPred2))
  println("Precision = " + evaluator2.evaluate(rfcvPred2))
  println("Recall = " + evaluator3.evaluate(rfcvPred2))
  println("F1 = " + evaluator4.evaluate(rfcvPred2))

  println("-------------------------------------------------------------------")
}

// COMMAND ----------

nbClass()

// COMMAND ----------

kmClass()

// COMMAND ----------

rfClass()

// COMMAND ----------

lsvcClass()

// COMMAND ----------

mpcClass()

// COMMAND ----------

nbClassCV()

// COMMAND ----------

kmClassCV()

// COMMAND ----------

rfClassCV()
