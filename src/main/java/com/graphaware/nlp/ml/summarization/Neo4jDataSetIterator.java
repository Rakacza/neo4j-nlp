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
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

import java.util.List;
import java.io.IOException;

/**
 * Neo4j data applyTransformToDestination iterator.
 * @author VlK
 */
public class Neo4jDataSetIterator extends BaseDatasetIterator {

    GraphDatabaseService database;

    /**Get the specified number of examples for training.
     * @param database Neo4j database
     * @param batch the batch size of the examples
     * @throws IOException
     */
    public Neo4jDataSetIterator(GraphDatabaseService database, List<Long> inputNodeIDs, int batch) {
        this(database, inputNodeIDs, batch, false, 0);
    }

    /**Get the specified number of MNIST examples (test or train set), with optional shuffling and binarization.
     * @param database Neo4j database
     * @param batch Size of each patch
     * @param shuffle whether to shuffle the examples
     * @param rndSeed random number generator seed to use when shuffling examples
     */
    public Neo4jDataSetIterator(GraphDatabaseService database, List<Long> inputNodeIDs, int batch, boolean shuffle, int rndSeed) {
        super(batch, 60000, new Neo4jDataFetcher(database, inputNodeIDs, shuffle, rndSeed));
    }

}