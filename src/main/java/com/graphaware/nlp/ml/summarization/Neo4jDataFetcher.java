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

import org.neo4j.graphdb.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
//import org.deeplearning4j.util.MathUtils;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Collections;
import java.util.Random;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Data fetcher for data in Neo4j database.
 * @author VlK
 *
 */
public class Neo4jDataFetcher extends BaseDataFetcher {
    private static final Logger LOG = LoggerFactory.getLogger(Neo4jDataFetcher.class);
    //private static final maxExamplesSize =
    private static final String idfPropertyName = "idfForSummarizer";
    private static final String BOW_QUERY = "MATCH (a:AnnotatedText)\n" +
            "where id(a) in {ids}\n" +
            "match (a)-[:CONTAINS_SENTENCE]->(:Sentence)-[r:HAS_TAG]->(t:Tag)\n" +
            //"with a, t, sum(r.tf) as tf, t.idfForSummarizer as idf\n" +
            //"order by id(a), tf desc, idf desc\n" +
            "with a, t, sum(r.tf) * t.idfForSummarizer as tfidf\n" +
            "order by id(a), tfidf desc\n" +
            "with a, collect(t.id)[..60] as BOW\n";
    private static final String mainQuery =  BOW_QUERY +
            "match (a)-[:CONTAINS_SENTENCE]->(s:Sentence)-[r:HAS_TAG]->(t:Tag)\n" +
            "where t.id in BOW\n" +
            "with a, BOW, s.sentenceNumber as sentenceNum, s.id as sentenceId, t, sum(r.tf) as tf\n" +
            "return id(a) as docId, BOW, sentenceNum, sentenceId, collect(t.id) as sentenceTags, collect(tf) as tfVals, collect(t.idfForSummarizer) as idfVals\n" +
            //"return id(a) as docId, BOW, sentenceNum, s.id as sentenceId, sentenceTags, extract(w in BOW | case when w in sentenceTags then 1 else 0 end)\n" +
            "order by docId, sentenceNum";

    protected GraphDatabaseService database;
    protected List<Long> inputNodeIDs;
    protected List<Long> currentNodeIDs;
    protected boolean shuffle;
    protected int rndSeed;
    protected Random rndGen;
    protected final int bowSize; // Bag-of-Words size

    public Neo4jDataFetcher(GraphDatabaseService database, List<Long> inputNodeIDs, boolean shuffle, int rndSeed) {
        this.database = database;
        this.inputNodeIDs = inputNodeIDs;
        this.currentNodeIDs = new ArrayList<>(inputNodeIDs);
        this.shuffle = shuffle;
        this.rndSeed = rndSeed;
        this.bowSize = 60;

        totalExamples = inputNodeIDs.size();
        numOutcomes = this.bowSize;
        inputColumns = this.bowSize;

        rndGen = new Random(rndSeed);
        reset(); //Shuffle order

        calculateAndStoreIdf();
    }

    @Override
    public void fetch(int numDocs) {
        LOG.info("Current batch size: " + numDocs);
        List<Long> ids = currentNodeIDs.stream().limit(numDocs).collect(Collectors.toList());
        currentNodeIDs = currentNodeIDs.stream().skip(numDocs).collect(Collectors.toList());
        cursor += ids.size();
        LOG.info("Actual number of documents: " + ids.size());

        Map<String, List<Float>> featureMatrix = new HashMap<>();
        Map<String, Integer> featureIndices = new HashMap<>();

        try (Transaction tx = database.beginTx();) {
            Map<String, Object> params = new HashMap<>();
            params.put("ids", ids);
            Result res = database.execute(mainQuery, params);
            int currIdx = 0;
            List<String> BOW = new ArrayList<>();
            Long docPrevious = -1L;
            while (res != null && res.hasNext()) {
                Map<String, Object> next = res.next();
                Long doc = (Long) next.get("docId");
                if (!doc.equals(docPrevious)) {
                    docPrevious = doc;
                    BOW.clear();
                    BOW = iterableToList((Iterable<String>) next.get("BOW"));
                    LOG.info("New BOW: document " + doc + ", BOW size " + BOW.size());
                    System.out.println(BOW);
                }
                if (BOW.size() < bowSize) {
                    LOG.info("Skipping current document: it is too short, current BOW size is " + BOW.size() + ", but " + bowSize + " is required. Aborting.");
                    continue;
                }
                int sentenceNum = (int) next.get("sentenceNum");
                String sentenceId = (String) next.get("sentenceId");
                List<String> sentenceTags = iterableToList((Iterable<String>) next.get("sentenceTags"));
                List<Integer> tfVals = iterableToList((Iterable<Integer>) next.get("tfVals"));
                List<Float> idfVals = iterableToList((Iterable<Float>) next.get("idfVals"));

                // Create sentence one-hot-encoding vector based on current document BOW
                LOG.info("Document " + doc + ", sentence " + sentenceNum + ", sentence.id " + sentenceId + ", tags: " + sentenceTags.stream().collect(Collectors.joining(", ")));
                List<Float> vec = createFeatureVector(sentenceTags, BOW);
                LOG.info("Vector: " + vec.stream().map(el -> el.toString()).collect(Collectors.joining(", ")));

                featureMatrix.put(sentenceId, vec);
                featureIndices.put(sentenceId, currIdx);
                currIdx += 1;
            }
            tx.success();
        } catch (Exception e) {
            LOG.error("Error while getting Bag of Words: ", e);
        }

        // Transform results into INDArray: 2 dimensions (rows = sentences, columns = one-hot-encoding vectors)
        float[][] featureData = new float[featureMatrix.size()][bowSize];
        float[][] labelData = new float[featureMatrix.size()][bowSize];
        for (String key: featureIndices.keySet()) {
            int n = featureMatrix.get(key).size();
            if (n != bowSize)
                LOG.error("Unexpected vector length: expected " + bowSize + ", found " + n);
            for (int i = 0; i < n; i++)
                featureData[featureIndices.get(key)][i] = featureMatrix.get(key).get(i);
        }
        featureMatrix.clear();
        featureIndices.clear();

        INDArray features = Nd4j.create(featureData);
        INDArray labels = Nd4j.create(labelData);
        curr = new DataSet(features, labels);
    }

    // TO DO: feature vector should contain tf or tf*idf values ?
    public static List<Float> createFeatureVector(List<String> tags, List<String> BOW) {
        List<Float> vec = new ArrayList<>();
        BOW.stream().forEach(el -> {
            if (tags.contains(el))
                vec.add(1.0f);
            else
                vec.add(0.0f);
        });
        return vec;
    }

    @Override
    public void reset() {
        cursor = 0;
        curr = null;
        if (shuffle)
            Collections.shuffle(inputNodeIDs, rndGen);
    }

    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }

    public static String getBowQuery() {
        return BOW_QUERY;
    }

    private void calculateAndStoreIdf() {
        LOG.info("Calculating idf values for Tags.");
        String query =
                "MATCH (a:AnnotatedText)-[:CONTAINS_SENTENCE]->(:Sentence)-[ht:HAS_TAG]->(t:Tag)\n"
                + "WHERE id(a) in {ids}\n"
                + "WITH t, log10(1.0f * {n_docs} / count(DISTINCT a)) as idf\n"
                + "SET t." + idfPropertyName + " = idf";
        Map<String, Object> params = new HashMap<>();
        params.put("ids", inputNodeIDs);
        params.put("n_docs", inputNodeIDs.size());
        try (Transaction tx = database.beginTx();) {
            database.execute(query, params);
            tx.success();
        } catch (Exception e) {
            LOG.error("Error while getting idf values: ", e);
        }
    }

    public static <T> List<T> iterableToList(Iterable<T> it) {
        List<T> newList = new ArrayList<>();
        for (T obj : it) {
            newList.add(obj);
        }
        return newList;
    }
}