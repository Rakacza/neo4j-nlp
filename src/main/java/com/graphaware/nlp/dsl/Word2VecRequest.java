/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.dsl;

import java.util.Map;
import org.neo4j.graphdb.Node;

public class Word2VecRequest {

    private static final String PARAMETER_NAME_ANNOTATED_TEXT = "node";
    private static final String PARAMETER_NAME_SPLIT_TAG = "splitTag";
    private static final String PARAMETER_NAME_FILTER_LANG = "filterLang";
    private static final String PARAMETER_NAME_LANG = "lang";
    private static final String PARAMETER_MODEL_NAME = "modelName";
    private static final String PARAMETER_PROPERTY_NAME = "propertyName";
    private static final String PARAMETER_PROPERTY_QUERY = "query";
    private static final String PARAMETER_NAME_TAG = "tag";
    private static final String PARAMETER_NAME_TEXT_PROCESSOR = "textProcessor";

    private static final String DEFAULT_PROPERTY_NAME = "word2vec";
    private final static String DEFAULT_LANGUAGE = "en";
    private static final boolean DEFAULT_SPLIT_TAG = false;
    private static final boolean DEFAULT_FILTER_LANG = true;

    private Node annotatedNode;
    private Node tagNode;
    private Boolean splitTags;
    private Boolean filterByLang;
    private String lang;
    private String query;
    private String modelName;
    private String propertyName;
    private String processor;

    public static Word2VecRequest fromMap(Map<String, Object> word2VecRankRequest) {
        Word2VecRequest result = new Word2VecRequest();
        result.setAnnotatedNode((Node) word2VecRankRequest.get(PARAMETER_NAME_ANNOTATED_TEXT));
        result.setTagNode((Node) word2VecRankRequest.get(PARAMETER_NAME_TAG));
        result.setSplitTags((Boolean) word2VecRankRequest.getOrDefault(PARAMETER_NAME_SPLIT_TAG, DEFAULT_SPLIT_TAG));
        result.setFilterByLang((Boolean) word2VecRankRequest.getOrDefault(PARAMETER_NAME_FILTER_LANG, DEFAULT_FILTER_LANG));
        result.setLang((String) word2VecRankRequest.getOrDefault(PARAMETER_NAME_LANG, DEFAULT_LANGUAGE));
        result.setQuery((String) word2VecRankRequest.get(PARAMETER_PROPERTY_QUERY));
        result.setModelName((String) word2VecRankRequest.get(PARAMETER_MODEL_NAME));
        result.setPropertyName((String) word2VecRankRequest.getOrDefault(PARAMETER_PROPERTY_NAME, DEFAULT_PROPERTY_NAME));
        result.setProcessor((String) word2VecRankRequest.getOrDefault(PARAMETER_NAME_TEXT_PROCESSOR, ""));
        return result;
    }

    public Node getAnnotatedNode() {
        return annotatedNode;
    }

    public void setAnnotatedNode(Node annotatedNode) {
        this.annotatedNode = annotatedNode;
    }

    public Boolean getSplitTags() {
        return splitTags;
    }

    public void setSplitTags(Boolean splitTags) {
        this.splitTags = splitTags;
    }

    public Boolean getFilterByLang() {
        return filterByLang;
    }

    public void setFilterByLang(Boolean filterByLang) {
        this.filterByLang = filterByLang;
    }

    public String getLang() {
        return lang;
    }

    public void setLang(String lang) {
        this.lang = lang;
    }

    public String getQuery() {
        return query;
    }

    public void setQuery(String query) {
        this.query = query;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getPropertyName() {
        return propertyName;
    }

    public void setPropertyName(String propertyName) {
        this.propertyName = propertyName;
    }

    public Node getTagNode() {
        return tagNode;
    }

    public void setTagNode(Node tagNode) {
        this.tagNode = tagNode;
    }

    public String getProcessor() {
        return processor;
    }

    public void setProcessor(String processor) {
        this.processor = processor;
    }
    
}