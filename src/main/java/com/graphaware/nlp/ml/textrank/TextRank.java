/*
 * Copyright (c) 2013-2016 GraphAware
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
package com.graphaware.nlp.ml.textrank;

import static com.graphaware.nlp.domain.Labels.Keyword;
import static com.graphaware.nlp.domain.Relationships.DESCRIBES;
import com.graphaware.nlp.util.MapUtil;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.stream.Collectors;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.ResourceIterator;
import org.neo4j.graphdb.Result;
import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class TextRank {

    private static final Logger LOG = LoggerFactory.getLogger(TextRank.class);
    private final GraphDatabaseService database;
    private boolean removeStopWords;
    private boolean directionMatters;
    private List<String> stopWords;
    private List<String> admittedPOSs;
    private Map<Long, String> idToValue = new HashMap<>();

    public TextRank(GraphDatabaseService database) {
        this.database = database;
        this.stopWords = Arrays.asList("new", "old", "large", "big", "small", "many", "few");
        this.removeStopWords = false;
        this.directionMatters = true;
        this.admittedPOSs = Arrays.asList("NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "Unknown"); // "Unknown" because we want to keep tags that have unknown POS
    }

    public void setStopwords(String stopwords) {
        this.stopWords = Arrays.asList(stopwords.split(",")).stream().map(str -> str.trim().toLowerCase()).collect(Collectors.toList());
        this.removeStopWords = true;
    }

    public void removeStopWords(boolean val) {
        this.removeStopWords = val;
    }

    public void respectDirections(boolean val) {
        this.directionMatters = val;
    }

    @Deprecated
    public boolean createCooccurrencesOld(Node annotatedText, String relType, String relWeight) {
        Map<String, Object> params = new HashMap<>();
        params.put("id", annotatedText.getId());
        params.put("relType", relType);
        params.put("relWeight", relWeight);
        try (Transaction tx = database.beginTx();) {
            database.execute("MATCH (a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)-[:SENTENCE_TAG_OCCURRENCE]->(to:TagOccurrence)\n"
                    + "WHERE a.id = {id} \n"
                    + "WITH s, to\n"
                    + "ORDER BY s.sentenceNumber, to.startPosition\n"
                    + "MATCH (to)-[:TAG_OCCURRENCE_TAG]->(t:Tag)\n"
                    + "WHERE size(t.value) > 2 AND ANY (p in t.pos where p in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'] )\n" // only nouns and adjectives
                    + "WITH s, collect(t) as tags\n"
                    + "UNWIND range(0, size(tags) - 2, 1) as i\n"
                    + "WITH s, tags[i] as tag1, tags[i+1] as tag2\n"
                    + "MERGE (tag1)-[c:" + relType + "]-(tag2)\n"
                    + "ON CREATE SET c.weight = 1\n"
                    + "ON MATCH SET c.weight = c.weight + 1\n",
                    params);
            tx.success();
        } catch (Exception e) {
            LOG.error("createCooccurrences() failed with QueryExecutionException: " + e.getMessage());
            return false;
        }
        return true;
    }

    public Map<Long, Map<Long, CoOccurrenceItem>> createCooccurrences(Node annotatedText) {
        Map<String, Object> params = new HashMap<>();
        params.put("id", annotatedText.getId());
        // a query for creating co-occurrences per sentence
        /*Result res = database.execute("MATCH (a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)-[:SENTENCE_TAG_OCCURRENCE]->(to:TagOccurrence)\n"
                + "WHERE id(a) = {id} \n"
                + "WITH s, to\n"
                + "ORDER BY s.sentenceNumber, to.startPosition\n"
                + "MATCH (to)-[:TAG_OCCURRENCE_TAG]->(t:Tag)\n"
                   // Only nouns and adjectives. Careful: sometimes POS property does not exist (mostly when NE is recognized)
                + "WHERE size(t.value) > 2\n"// AND ( ANY (p in t.pos where p in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']) OR NOT EXISTS(t.pos) )\n"
                + "WITH s, collect(t) as tags\n"
                + "UNWIND range(0, size(tags) - 2, 1) as i\n"
                + "RETURN s, id(tags[i]) as tag1, id(tags[i+1]) as tag2, tags[i].value as tag1_val, tags[i+1].value as tag2_val, "
                    + "(case tags[i].pos when null then 'Unknown,' else reduce(pos='', p in tags[i].pos | pos+p+',') end) as pos1, (case tags[i+1].pos when null then 'Unknown,' else reduce(pos='', p in tags[i+1].pos | pos+p+',') end) as pos2\n",
                params);*/
        // this is based on orignal TextRank: a query that doesn't care about sentence boundaries (it connects last word of a sentence with 1st word of the next sentence)
        Result res = database.execute("MATCH (a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)-[:SENTENCE_TAG_OCCURRENCE]->(to:TagOccurrence)\n"
                + "WHERE id(a) = {id} \n"
                + "WITH to\n"
                + "ORDER BY to.startPosition\n"
                + "MATCH (to)-[:TAG_OCCURRENCE_TAG]->(t:Tag)\n"
                   // Careful: sometimes POS property does not exist (mostly when NE is recognized)
                + "WHERE size(t.value) > 2\n" //AND ( ANY (p in t.pos where p in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']) OR NOT EXISTS(t.pos) )\n"
                + "WITH collect(t) as tags\n"
                + "UNWIND range(0, size(tags) - 2, 1) as i\n"
                + "RETURN id(tags[i]) as tag1, id(tags[i+1]) as tag2, tags[i].value as tag1_val, tags[i+1].value as tag2_val, "
                    + "(case tags[i].pos when null then 'Unknown,' else reduce(pos='', p in tags[i].pos | pos+p+',') end) as pos1, (case tags[i+1].pos when null then 'Unknown,' else reduce(pos='', p in tags[i+1].pos | pos+p+',') end) as pos2\n",
                params);
        Map<Long, Map<Long, CoOccurrenceItem>> results = new HashMap<>();
        while (res != null && res.hasNext()) {
            Map<String, Object> next = res.next();
            Long tag1 = (Long) next.get("tag1");
            Long tag2 = (Long) next.get("tag2");
            List<String> pos1 = Arrays.asList( ((String) next.get("pos1")).split(",") );
            List<String> pos2 = Arrays.asList( ((String) next.get("pos2")).split(",") );

            // check whether POS of both tags are admitted
            if (pos1.stream().filter(pos -> pos!=null && admittedPOSs.contains(pos)).count()==0)
                continue;
            if (pos2.stream().filter(pos -> pos!=null && admittedPOSs.contains(pos)).count()==0)
                continue;

            // fill tag co-occurrences (adjacency matrix)
            addTagToCoOccurrence(results, tag1, tag2);
            if (!directionMatters) // when direction of co-occurrence relationships is not important
              addTagToCoOccurrence(results, tag2, tag1);

            // for logging purposses
            idToValue.put(tag1, (String) next.get("tag1_val"));
            idToValue.put(tag2, (String) next.get("tag2_val"));
            LOG.debug("Adding co-occurrence: " + (String) next.get("tag1_val") + " -> " + (String) next.get("tag2_val"));
        }
        return results;
    }

    private void addTagToCoOccurrence(Map<Long, Map<Long, CoOccurrenceItem>> results, Long tag1, Long tag2) {
        Map<Long, CoOccurrenceItem> mapTag1;
        if (!results.containsKey(tag1)) {
            mapTag1 = new HashMap<>();
            results.put(tag1, mapTag1);
        } else {
            mapTag1 = results.get(tag1);
        }
        if (mapTag1.containsKey(tag2)) {
            mapTag1.get(tag2).incCount();
        } else {
            mapTag1.put(tag2, new CoOccurrenceItem(tag1, tag2));
        }
    }

    @Deprecated
    public boolean deleteCooccurrences(Long annotatedID, String relType) {
        Map<String, Object> params = new HashMap<>();
        params.put("id", annotatedID);
        try (Transaction tx = database.beginTx();) {
            database.execute("MATCH (a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)\n"
                    + "WHERE a.id = {id}\n"
                    + "MATCH (s)-[:HAS_TAG]->(:Tag)-[co:" + relType + "]->(:Tag)\n"
                    + "DELETE co",
                    params);
            tx.success();
        } catch (Exception e) {
            LOG.error("deleteCooccurrences() failed with QueryExecutionException: " + e.getMessage());
            return false;
        }
        return true;
    }

    public boolean evaluate(Node annotatedText, Map<Long, Map<Long, CoOccurrenceItem>> coOccurrence, int iter, double damp, double threshold) {
        PageRank pageRank = new PageRank(database);
        Map<Long, Double> pageRanks = pageRank.run(coOccurrence, iter, damp, threshold);
        List<Long> top10 = getTop10(pageRanks);

        pageRanks.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
            .forEach(en -> LOG.info(idToValue.get(en.getKey()) + ": " + en.getValue()));
        LOG.info("Sum of PageRanks = " + pageRanks.values().stream().mapToDouble(Number::doubleValue).sum());
        String topStr = "";
        for (Long id: top10) {
            topStr += idToValue.get(id) + ", ";
        }
        LOG.info("Top 10 tags: " + topStr);

        Map<String, Object> params = new HashMap<>();
        params.put("id", annotatedText.getId());
        params.put("nodeList", top10);
        Result res = database.execute(
                "MATCH (node:Tag)<-[:TAG_OCCURRENCE_TAG]-(to:TagOccurrence)<-[:SENTENCE_TAG_OCCURRENCE]-(:Sentence)<-[:CONTAINS_SENTENCE]-(a:AnnotatedText)\n"
                + "WHERE id(a) = {id} and id(node) in {nodeList}\n"
                + "RETURN node.id as tag, to.startPosition as sP, to.endPosition as eP, id(node) as tagId\n"
                + "ORDER BY sP asc",
                params);

        // First: merge neighboring words into phrases
        long prev_eP = -1000;
        String keyword = "";
        Map<String, Integer> results = new HashMap<>();
        Map<String, Double> keywords = new HashMap<>();
        while (res != null && res.hasNext()) {
            Map<String, Object> next = res.next();
            int startPosition = (int) next.get("sP");
            int endPosition = (int) next.get("eP");
            Long tagId = (Long) next.get("tagId");
            
            Double score = pageRanks.get(tagId);
            String tag = (String) next.get("tag");
            
            
            final String[] tagSplit = tag.split("_");
            if (tagSplit.length > 2) {
                LOG.warn("Tag " + tag + " has more than 1 underscore symbols");
            }
            
            String tagVal = tagSplit[0];
            String lang = tagSplit[1];

            if (removeStopWords && stopWords.stream().anyMatch(str -> str.equals(tagVal))) {
                continue;
            }
            keywords.put(tag, score);
            
            if (startPosition - prev_eP <= 1) {
                keyword += " " + tagVal;
            } else {
                if (keyword.split(" ").length > 1) {
                    if (results.containsKey(keyword + "_" + lang)) {
                        results.put(keyword + "_" + lang, results.get(keyword + "_" + lang) + 1);
                    } else {
                        results.put(keyword + "_" + lang, 1);
                    }
                }
                keyword = tagVal;
            }
            prev_eP = endPosition;
            
        }

        // Next: include into final result simply top-x single words
        keywords.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                .limit(8)
                .forEach(en -> {
                    results.put(en.getKey(), 1);
                    LOG.debug(en.getKey());
                });

        LOG.info("--- Results: ");
        results.keySet().stream().map((key) -> {
            if (key.split("_").length > 2) {
                LOG.warn("Tag " + key + " has more than 1 underscore symbols, newly created Keyword node might be wrong");
            }
            return key;
        }).forEachOrdered((key) -> {
            String val = key.split("_")[0];
            if (!(removeStopWords && stopWords.stream().anyMatch(str -> str.equals(val)))) {
                Node newNode;
                ResourceIterator<Node> findNodes = database.findNodes(Keyword, "id", key);
                if (findNodes.hasNext()) {
                    newNode = findNodes.next();
                } else {
                    newNode = database.createNode(Keyword);
                    newNode.setProperty("id", key);
                    newNode.setProperty("value", val);
                }
                if (newNode != null) {
                    Relationship rel = newNode.createRelationshipTo(annotatedText, DESCRIBES);
                    rel.setProperty("count", results.get(key));
                }
                LOG.info(val);
            }
        });

        return true;
    }

    private List<Long> getTop10(Map<Long, Double> pageRanks) {
        /*List<Map.Entry<Long, Double>> sortedPageRanks = MapUtil.getSortedListByValue(pageRanks);
        int maxTop = pageRanks.size() > 10 ? 10 : pageRanks.size();
        List<Long> top10 = sortedPageRanks.subList(0, maxTop).stream()
                .map((item) -> item.getKey()).collect(Collectors.toList());*/
        List<Long> top10 = pageRanks.entrySet().stream()
            .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
            .limit(10)
            .map((item) -> item.getKey()).collect(Collectors.toList());
        return top10;
    }
}