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
package com.graphaware.nlp.dsl.procedure;

import com.graphaware.nlp.dsl.AbstractDSL;
import com.graphaware.nlp.dsl.request.SummarizationRequest;
import com.graphaware.nlp.dsl.result.SingleResult;
import com.graphaware.nlp.ml.summarization.SummarizationProcessor;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.Map;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SummarizationProcedure extends AbstractDSL {

    private static final Logger LOG = LoggerFactory.getLogger(SummarizationProcedure.class);

    @Procedure(name = "ga.nlp.ml.summarize.train", mode = Mode.WRITE)
    @Description("Summarization procedure")
    public Stream<SingleResult> train(@Name("summarizationRequest") Map<String, Object> summarizationRequest) {
        try {
            SummarizationRequest request = SummarizationRequest.fromMap(summarizationRequest);
            SummarizationProcessor processor = (SummarizationProcessor) getNLPManager().getExtension(SummarizationProcessor.class);
            return Stream.of(processor.train(request));
        } catch (Exception e) {
            LOG.error("ERROR in Summarization", e);
            throw new RuntimeException(e);
        }
    }

    @Procedure(name = "ga.nlp.ml.summarize", mode = Mode.WRITE)
    @Description("Summarization procedure")
    public Stream<SingleResult> summarize(@Name("summarizationRequest") Map<String, Object> summarizationRequest) {
        try {
            SummarizationRequest request = SummarizationRequest.fromMap(summarizationRequest);
            SummarizationProcessor processor = (SummarizationProcessor) getNLPManager().getExtension(SummarizationProcessor.class);
            return Stream.of(processor.process(request));
        } catch (Exception e) {
            LOG.error("ERROR in Summarization", e);
            throw new RuntimeException(e);
        }
    }

}
