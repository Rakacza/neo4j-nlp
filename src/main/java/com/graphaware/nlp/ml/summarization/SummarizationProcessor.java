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

import com.graphaware.nlp.annotation.NLPModuleExtension;
import com.graphaware.nlp.dsl.request.SummarizationRequest;
import com.graphaware.nlp.dsl.result.SingleResult;
import com.graphaware.nlp.extension.AbstractExtension;
import com.graphaware.nlp.extension.NLPExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@NLPModuleExtension(name = "SummarizationProcessor")
public class SummarizationProcessor extends AbstractExtension implements NLPExtension {

    private static final Logger LOG = LoggerFactory.getLogger(SummarizationProcessor.class);

    public SingleResult train(SummarizationRequest request) {
        ExtractiveSummarization.Builder summarizerBuilder = new ExtractiveSummarization.Builder(getDatabase(), getNLPManager().getConfiguration());
        summarizerBuilder.setTrainingIterations(request.getTrainingIterations())
                .setTrainingEpochs(request.getTrainingEpochs())
                .setTrainingBatchSize(request.getTrainingBatchSize());

        ExtractiveSummarization summarizer = summarizerBuilder.build();
        summarizer.train(request.getTrainingIDs());

        LOG.info("Trainig procedure completed.");
        //return res ? SingleResult.success() : SingleResult.fail();
        return SingleResult.success();
    }

    public SingleResult process(SummarizationRequest request) {
        ExtractiveSummarization.Builder summarizerBuilder = new ExtractiveSummarization.Builder(getDatabase(), getNLPManager().getConfiguration());

        ExtractiveSummarization summarizer = summarizerBuilder.build();
        summarizer.run(request.getNode());

        LOG.info("AnnotatedText with ID " + request.getNode().getId() + " processed.");
        //return res ? SingleResult.success() : SingleResult.fail();
        return SingleResult.success();
    }

}
