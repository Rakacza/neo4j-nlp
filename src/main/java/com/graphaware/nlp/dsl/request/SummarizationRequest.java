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
package com.graphaware.nlp.dsl.request;

import org.neo4j.graphdb.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.List;
import java.util.ArrayList;

public class SummarizationRequest {

    private static final Logger LOG = LoggerFactory.getLogger(SummarizationRequest.class);

    private final static String PARAMETER_ANNOTATED_TEXT = "annotatedText";
    private final static String PARAMETER_TRAIN_IDS = "annotatedTextIDs";
    private final static String PARAMETER_TRAIN_ITERATIONS = "trainingIterations";
    private final static String PARAMETER_TRAIN_EPOCHS = "trainingEpochs";
    private final static String PARAMETER_TRAIN_BATCH_SIZE = "trainingBatchSize";

    private final static int DEFAULT_TRAIN_ITERATIONS = 1;
    private final static int DEFAULT_TRAIN_EPOCHS = 1;
    private final static int DEFAULT_TRAIN_BATCH_SIZE = 50;

    private Node node;
    private List<Long> trainingIDs;
    private int trainingIterations;
    private int trainingEpochs;
    private int trainingBatch;

    public static SummarizationRequest fromMap(Map<String, Object> summarizationRequest) {
        SummarizationRequest result = new SummarizationRequest();
        result.setNode((Node) summarizationRequest.getOrDefault(PARAMETER_ANNOTATED_TEXT, null));
        result.setTrainingIDs(iterableToList((Iterable<Long>) summarizationRequest.getOrDefault(PARAMETER_TRAIN_IDS, null)));
        result.setTrainingIterations(((Number) summarizationRequest.getOrDefault(PARAMETER_TRAIN_ITERATIONS, DEFAULT_TRAIN_ITERATIONS)).intValue());
        result.setTrainingEpochs(((Number) summarizationRequest.getOrDefault(PARAMETER_TRAIN_EPOCHS, DEFAULT_TRAIN_EPOCHS)).intValue());
        result.setTrainingBatchSize(((Number) summarizationRequest.getOrDefault(PARAMETER_TRAIN_BATCH_SIZE, DEFAULT_TRAIN_BATCH_SIZE)).intValue());

        return result;
    }

    public Node getNode() {
        return node;
    }

    public void setNode(Node node) {
        this.node = node;
    }

    public List<Long> getTrainingIDs() {
        return trainingIDs;
    }

    public void setTrainingIDs(List<Long> ids) {
        this.trainingIDs = ids;
    }

    public int getTrainingIterations() {
        return trainingIterations;
    }

    public void setTrainingIterations(int iter) {
        this.trainingIterations = iter;
    }

    public int getTrainingEpochs() {
        return trainingEpochs;
    }

    public void setTrainingEpochs(int i) {
        this.trainingEpochs = i;
    }

    public int getTrainingBatchSize() {
        return trainingBatch;
    }

    public void setTrainingBatchSize(int i) {
        this.trainingBatch = i;
    }

    private static <T> List<T> iterableToList(Iterable<T> it) {
        List<T> newList = new ArrayList<>();
        if (it == null)
            return newList;
        for (T obj : it) {
            newList.add(obj);
        }
        return newList;
    }
}
