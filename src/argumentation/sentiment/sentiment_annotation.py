import os
from xml.etree.ElementTree import tostring, Element, SubElement
from xml.dom import minidom


def prepare_bert_trained_dataset(data, loc):
    '''
    Convert data from hugging face to xml format

    Args:
    data: data from hugging face
    loc: location for the xml file

    '''
    root = Element('reviews')
    for index, review in enumerate(data):
        id_node = SubElement(root, 'sentenceId')
        id_node.text = review['sentenceId']
        sentence_node = SubElement(root, 'sentences')
        text_node = SubElement(sentence_node, 'text')
        text_node.text = review['text']
        aspects_node = SubElement(sentence_node, 'aspectTerms')
        # if there are aspect in aspect terms
        aspectTerms = review['aspectTerms']
        if len(aspectTerms) > 0:
            for term in aspectTerms:
                aspect_node = SubElement(aspects_node, 'term')
                aspect_node.text = term['term']
                aspect_node.attrib['from'] = term['from']
                aspect_node.attrib['to'] = term['to']
                aspect_node.attrib['polarity'] = term['polarity']

    xmlstr = minidom.parseString(tostring(root)).toprettyxml(indent='   ')
    xmlstr = os.linesep.join([s for s in xmlstr.splitlines() if s.strip()])
    with open(loc, 'w') as f:
        f.write(xmlstr)
