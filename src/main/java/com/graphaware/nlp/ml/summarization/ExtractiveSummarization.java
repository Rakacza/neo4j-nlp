/*
 * Copyright (c) 2013-2017 GraphAware
 *
 * This file is part of the GraphAware Framework.
 *
 * GraphAware Framework is free software: you can redistribute it and/or modify it under the terms of
 * the GNU General Public License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details. You should have received a copy of
 * the GNU General Public License along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */
package com.graphaware.nlp.ml.summarization;

import com.graphaware.nlp.configuration.DynamicConfiguration;
import com.graphaware.nlp.ml.similarity.CosineSimilarity;
//import com.graphaware.nlp.domain.TfIdfObject;
//import org.bytedeco.javacv.FrameFilter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.neo4j.graphdb.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.stream.Collectors;
//import java.util.concurrent.atomic.AtomicReference;

public class ExtractiveSummarization {

    private static final Logger LOG = LoggerFactory.getLogger(ExtractiveSummarization.class);
    private static final String PATH_TO_MODEL = "import/summarization--trained-model.zip";
    private static final String REFERENCE_QUERY = "match (n:TestEmail)-[:HAS_ANNOTATED_TEXT]->(a:AnnotatedText) where id(a) = {id} with n.subject as text\n";
    private final int trainingIterations;
    private final int trainingEpochs;
    private final int trainingBatchSize;

    private final GraphDatabaseService database;

    public ExtractiveSummarization(GraphDatabaseService database,
                                   int iterations,
                                   int epochs,
                                   int batches) {
        this.database = database;
        this.trainingIterations = iterations;
        this.trainingEpochs = epochs;
        this.trainingBatchSize = batches;
    }

    public void run(Node annotatedText) {
        System.out.println("Starting the run() method.");

        MultiLayerNetwork model = loadModel();
        if (model == null)
            return;
        //System.out.println("Printing configurations:");
        //model.printConfiguration();
        //System.out.println(">>> Parameters: " + model.params().length());

        LOG.info(">>> Processing annotated text " + annotatedText.getId());
        // Retrieve data: `inputData` is a 2D array with rows being sentences represented as vectors
        INDArray inputData = inputDataToVectors(annotatedText, "");
        INDArray inputReference = inputDataToVectors(annotatedText, REFERENCE_QUERY);
        if (inputData.length() == 0 || inputReference.length() == 0)
            return;

        // Get compressed feature vectors for inputData
        int layerNumber = 3;
        List<INDArray> propagated_layer = model.feedForwardToLayer(layerNumber, inputData);
        System.out.format("Compressed feature vector - data: n_rows = %d, n_columns = %d\n", propagated_layer.get(layerNumber + 1).rows(), propagated_layer.get(layerNumber + 1).columns());
        //System.out.println("\n --- " + propagated_layer.get(4).getDouble(0, 0));
        INDArray inputData_propagated = propagated_layer.get(layerNumber + 1);
        propagated_layer.clear();

        // Get compressed feature vectors for inputReference
        propagated_layer = model.feedForwardToLayer(layerNumber, inputReference);
        System.out.format("\nCompressed feature vector - reference: n_rows = %d, n_columns = %d\n", propagated_layer.get(layerNumber + 1).rows(), propagated_layer.get(layerNumber + 1).columns());
        INDArray inputReference_propagated = propagated_layer.get(layerNumber + 1);
        propagated_layer.clear();

        // For testing the auto-encoder model by comparing input and output
        List<INDArray> prediction = model.feedForwardToLayer(model.getLayers().length - 1, inputData);

        LOG.info(">>> Results");
        System.out.println();
        for (int i = 0; i < inputData_propagated.rows(); i++)
            System.out.println("Row " + i + " - original vectors: cosine_similarity(wrt inputReference) = " + cosineSimilarity(inputReference, inputData.getRow(i)));
        System.out.println();
        for (int i = 0; i < inputData_propagated.rows(); i++)
            System.out.println("Row " + i + " - auto-encoder quality test: cosineSimilarity(input, output) = " + cosineSimilarity(inputData.getRow(i), prediction.get(model.getLayers().length).getRow(i)));
        System.out.println();
        for (int i = 0; i < inputData_propagated.rows(); i++)
            System.out.println("Row " + i + ": cosine_similarity(wrt inputReference) = " + cosineSimilarity(inputReference_propagated, inputData_propagated.getRow(i)));
        System.out.println();
    }

    public void train(List<Long> inputNodeIDs) {
        LOG.info("Starting the training procedure.");
        int seed = 123;
        int inputDim = 60;
        int hidden1 = 140;
        int hidden2 = 40;
        int hidden3 = 30;
        int hidden4 = 10;

        LOG.info("Number of input AnnotatedText IDs: " + inputNodeIDs.size());
        if (inputNodeIDs.size() == 0)
            return;

        // Configure neural network model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(trainingIterations)
                //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                //.learningRate(0.05)
                //.regularization(true).l2(0.0001)
                .list()
                // LossFunctions.LossFunction.MSE, KL_DIVERGENCE, NEGATIVELOGLIKELIHOOD
                .layer(0, new RBM.Builder().nIn(inputDim).nOut(hidden1).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(1, new RBM.Builder().nIn(hidden1).nOut(hidden2).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(2, new RBM.Builder().nIn(hidden2).nOut(hidden3).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(3, new RBM.Builder().nIn(hidden3).nOut(hidden4).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build()) //encoding stops
                .layer(4, new RBM.Builder().nIn(hidden4).nOut(hidden3).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build()) //decoding starts
                .layer(5, new RBM.Builder().nIn(hidden3).nOut(hidden2).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(6, new RBM.Builder().nIn(hidden2).nOut(hidden1).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(hidden1).nOut(inputDim).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .pretrain(true).backprop(true)
                //.pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(1));

        LOG.info(" >>> Loading data ...");
        DataSetIterator iter = new Neo4jDataSetIterator(database, inputNodeIDs, trainingBatchSize);

        LOG.info(" >>> Training model ...");
        for (int i = 0; i < trainingEpochs; i++) {
            LOG.info("Starting epoch {} of {}", (i + 1), trainingEpochs);
            while (iter.hasNext()) {
                LOG.info("   >> New batch");
                DataSet next = iter.next();
                model.fit(new DataSet(next.getFeatureMatrix(), next.getFeatureMatrix()));
            }
            iter.reset();
        }

        LOG.info(" >>> Saving the trained model ...");
        File locationToSave = new File(PATH_TO_MODEL); //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true; //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        try {
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        } catch(IOException e) {
            LOG.error("Failed to save the trained model!", e);
        }

    }

    private MultiLayerNetwork loadModel() {
        LOG.info(">>> Loading neural network model ...");
        MultiLayerNetwork model;
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(PATH_TO_MODEL);
        } catch (Exception e) {
            LOG.error("Error loading neural network model from path " + PATH_TO_MODEL, e);
            return null;
        }
        return model;
    }

    private INDArray inputDataToVectors(Node annotatedText, String query) {
        INDArray inputData = Nd4j.create(new float[0]);
        if (annotatedText == null) {
            LOG.error("Provided AnnotatedText is null, aborting!");
            return inputData;
        }

        List<Long> inputList = new ArrayList<>();
        inputList.add(annotatedText.getId());

        if (query.isEmpty()) {
            // Retrieve data: `inputData` is a 2D array with rows being sentences represented as vectors
            DataSetIterator iter = new Neo4jDataSetIterator(database, inputList, 1);
            if (!iter.hasNext()) {
                LOG.error("Data not retrieved, aborting!");
                return inputData;
            }
            inputData = iter.next().getFeatureMatrix();
            LOG.info("Input data: n_rows = " + inputData.rows() + ", n_columns = " + inputData.columns());
        } else {
            // First: get Bag-of-Words
            List<String> BOW = new ArrayList<>();
            String q = Neo4jDataFetcher.getBowQuery() + " return BOW";
            try (Transaction tx = database.beginTx();) {
                Map<String, Object> params = new HashMap<>();
                params.put("ids", inputList);
                Result result = database.execute(q, params);
                if (result != null && result.hasNext()) {
                    Map<String, Object> next = result.next();
                    BOW = Neo4jDataFetcher.iterableToList((Iterable<String>) next.get("BOW"));
                }
                tx.success();
            } catch (Exception e) {
                LOG.error("Error while getting Bag-of-Words: ", e);
                return inputData;
            }
            System.out.println(" > BOW: " + BOW);

            // Second: get vector representation of sentences using current Bag-of-Words
            String finalQuery = query + "with ga.nlp.processor.annotate(toLower(text), {name: \"tokenizer\"}) as annotated\n"
                + "with keys(annotated.sentences[0].tagOccurrences) as keys, annotated\n"
                + "return extract(k in keys | annotated.sentences[0].tagOccurrences[toString(k)][0].element.id) as tags";
            try (Transaction tx = database.beginTx();) {
                Map<String, Object> params = new HashMap<>();
                params.put("id", annotatedText.getId());
                Result result = database.execute(finalQuery, params);
                if (result != null && result.hasNext()) {
                    Map<String, Object> next = result.next();
                    List<String> tags = Neo4jDataFetcher.iterableToList((Iterable<String>) next.get("tags"));
                    //List<String> BOW = Neo4jDataFetcher.iterableToList((Iterable<String>) next.get("BOW"));

                    List<Float> vector = Neo4jDataFetcher.createFeatureVector(tags, BOW.stream().map(el -> el.toLowerCase()).collect(Collectors.toList()));
                    float[] vector_as_array = new float[vector.size()];
                    for (int i = 0; i < vector.size(); i++)
                        vector_as_array[i] = vector.get(i);
                    inputData = Nd4j.create(vector_as_array);
                    LOG.info("Reference input data: n_rows = " + inputData.rows() + ", n_columns = " + inputData.columns());
                }
                tx.success();
            } catch (Exception e) {
                LOG.error("Error while annotating summarization reference query: ", e);
                return inputData;
            }
        }

        //System.out.println("Input, dim0: " + inputData.size(0));
        //System.out.println("Input, dim1: " + inputData.size(1));
        System.out.println(inputData + " -> sum = " + inputData.sum(1));

        return inputData;
    }

    private double cosineSimilarity(INDArray vec1, INDArray vec2) {
        double sim = -1.;
        if (vec1.rows() != 1 || vec2.rows() != 1) {
            LOG.error("Wrong dimension of input vectors, aborting.");
            return sim;
        }
        CosineSimilarity simClass = new CosineSimilarity();
        sim = simClass.getSimilarity(indarrayToList(vec1), indarrayToList(vec2));
        return sim;
    }

    private static List<Double> indarrayToList(INDArray arr) {
        List<Double> res = new ArrayList<>();
        for (int i = 0; i < arr.columns(); i++)
            res.add(arr.getDouble(0, i));
        return res;
    }


    public static class Builder {

        private final GraphDatabaseService database;
        private static final int DEFAULT_TRAINING_ITERATIONS = 1;
        private static final int DEFAULT_TRAINING_EPOCHS = 1;
        private static final int DEFAULT_TRAINING_BATCH_SIZE = 50;

        private int trainingIterations = DEFAULT_TRAINING_ITERATIONS;
        private int trainingEpochs = DEFAULT_TRAINING_EPOCHS;
        private int trainingBatchSize = DEFAULT_TRAINING_BATCH_SIZE;

        public Builder(GraphDatabaseService database, DynamicConfiguration configuration) {
            this.database = database;
        }

        public ExtractiveSummarization build() {
            ExtractiveSummarization result = new ExtractiveSummarization(database,
                    trainingIterations,
                    trainingEpochs,
                    trainingBatchSize);
            return result;
        }

        public Builder setTrainingIterations(int iter) {
            this.trainingIterations = iter;
            return this;
        }

        public Builder setTrainingEpochs(int i) {
            this.trainingEpochs = i;
            return this;
        }

        public Builder setTrainingBatchSize(int i) {
            this.trainingBatchSize = i;
            return this;
        }
    }
}
