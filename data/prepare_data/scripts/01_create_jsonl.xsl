<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="3.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema"
  xmlns:sal="https://www.salamanca.school/schema"
  xmlns:tei="http://www.tei-c.org/ns/1.0"
  xpath-default-namespace="http://www.tei-c.org/ns/1.0"
  exclude-result-prefixes="xs sal tei">

  <!-- ============================================================
       Parameters
       ============================================================ -->
  <xsl:param name="output-format" select="'jsonl'"/>

  <!-- ============================================================
       Output setup
       ============================================================ -->
  <xsl:output method="text" encoding="UTF-8"/>

  <!-- ============================================================
       Helper functions / variables
       ============================================================ -->

  <xsl:function name="sal:json-escape" as="xs:string">
    <xsl:param name="s" as="xs:string"/>
    <xsl:value-of select="
      replace(
        replace(
          replace($s, '\\', '\\\\'),
          '&quot;', '\\&quot;'
        ),
        '&#10;', '\\n'
      )
    "/>
  </xsl:function>

  <xsl:function name="sal:csv-field" as="xs:string">
    <xsl:param name="s" as="xs:string"/>
    <xsl:value-of select="concat('&quot;', replace($s, '&quot;', '&quot;&quot;'), '&quot;')"/>
  </xsl:function>

  <xsl:key name="chars" match="tei:char" use="@xml:id"/>
  <xsl:variable name="_charDecl">
    <tei:charDecl xml:id="charDecl" xmlns="http://www.tei-c.org/ns/1.0">

<!-- Characters below 00FF -->

        <char xml:id="char0026">
            <desc>AMPERSAND</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>amp</value>
            </charProp>
            <mapping type="precomposed">&amp;</mapping>
            <mapping type="standardized">&amp;</mapping>
        </char>
    
        <char xml:id="char00b6">
            <desc>PILCROW SIGN</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>para</value>
            </charProp>
            <mapping type="precomposed">¶</mapping>
            <mapping type="standardized">¶</mapping>
        </char>
        <char xml:id="charf1e1">
            <desc>PARAGRAPHUS</desc>
        </char>
    
        <char xml:id="char00df">
            <desc>LATIN SMALL LETTER SHARP S</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>szlig</value>
            </charProp>
            <mapping type="precomposed">ß</mapping>
            <mapping type="standardized">ss</mapping>
        </char>

        <char xml:id="char00e6">
            <desc>LATIN SMALL LETTER AE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>aelig</value>
            </charProp>
            <mapping type="MUFI" subtype="Lat1Suppl">U+00E6</mapping>
            <mapping type="precomposed">æ</mapping>
            <mapping type="standardized">ae</mapping>
        </char>
        <char xml:id="char00c6">
            <desc>LATIN CAPITAL LETTER AE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>AElig</value>
            </charProp>
            <mapping type="MUFI" subtype="Lat1Suppl">U+00C6</mapping>
            <mapping type="precomposed">Æ</mapping>
            <mapping type="standardized">AE</mapping>
        </char>
        
        <char xml:id="chara0302">
            <desc>LATIN SMALL LETTER A WITH CIRCUMFLEX</desc>
            <mapping type="precomposed">â</mapping>
            <mapping type="composed">â</mapping>
            <mapping type="standardized">a</mapping>
        </char>
        
        <char xml:id="chara0300">
            <desc>LATIN SMALL LETTER A WITH GRAVE</desc>
            <mapping type="precomposed">à</mapping>
            <mapping type="composed">à</mapping>
            <mapping type="standardized">a</mapping>
        </char>

<!-- Other Precomposed Characters -->

        <char xml:id="char0180">
            <desc>LATIN SMALL LETTER B WITH STROKE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>bstrok</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtB">U+0180</mapping>
            <mapping type="precomposed">ƀ</mapping>
            <mapping type="standardized">b</mapping>
        </char>
        <char xml:id="char0111">
            <desc>LATIN SMALL LETTER D WITH STROKE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>dstrok</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+0111</mapping>
            <mapping type="precomposed">đ</mapping>
            <mapping type="standardized">d</mapping>
        </char>
        <char xml:id="char0142">
            <desc>LATIN SMALL LETTER L WITH STROKE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>lstrok</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+0142</mapping>
            <mapping type="precomposed">ł</mapping>
            <mapping type="standardized">l</mapping>
        </char>
        <char xml:id="char0153">
            <desc>LATIN SMALL LIGATURE OE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>oelig</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+0153</mapping>
            <mapping type="precomposed">œ</mapping>
            <mapping type="standardized">oe</mapping>
        </char>
        <char xml:id="char0152">
            <desc>LATIN CAPITAL LIGATURE OE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>Oelig</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+0152</mapping>
            <mapping type="precomposed">Œ</mapping>
            <mapping type="standardized">OE</mapping>
        </char>
        
        <char xml:id="char017f">
            <desc>LATIN SMALL LETTER LONG S</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>slong</value>
            </charProp>
            <mapping type="precomposed">ſ</mapping>
            <mapping type="standardized">s</mapping>
        </char>
        <char xml:id="char1e9c">
            <desc>LATIN SMALL LETTER LONG S WITH DIAGONAL STROKE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>slongbarslash</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtAdd"/>
            <mapping type="precomposed">ẜ</mapping>
            <mapping type="standardized">s</mapping>
        </char>
        <char xml:id="char01d4">
            <desc>LATIN SMALL LETTER U WITH CARON</desc>
            <mapping type="precomposed">ǔ</mapping>
            <mapping type="standardized">u</mapping>
        </char>
        <char xml:id="charu0300">
            <desc>LATIN SMALL LETTER U WITH GRAVE</desc>
            <mapping type="precomposed">ù</mapping>
            <mapping type="composed">ù</mapping>
            <mapping type="standardized">u</mapping>
        </char>
        <char xml:id="char0250">
            <desc>LATIN SMALL LETTER TURNED A</desc>
            <mapping type="precomposed">ɐ</mapping>
            <mapping type="standardized">a</mapping>
        </char>
        <char xml:id="char0259">
            <desc>LATIN SMALL LETTER SCHWA</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>schwa</value>
            </charProp>
            <mapping type="precomposed">ə</mapping>
            <mapping type="standardized">e</mapping>
        </char>
        <char xml:id="char026f">
            <desc>LATIN SMALL LETTER TURNED M</desc>
            <charProp>
                <localName>entity</localName>
                <value>turnedm</value>
            </charProp>
            <mapping type="precomposed">ɯ</mapping>
            <mapping type="standardized">m</mapping>
        </char>
        <char xml:id="char0292">
            <desc>LATIN SMALL LETTER EZH</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>ezh</value>
            </charProp>
            <mapping type="precomposed">ʒ</mapping>
            <mapping type="standardized">z</mapping>
        </char>
        <char xml:id="char204a">
            <desc>LATIN ABBREVIATION SIGN SMALL ET</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>et</value>
            </charProp>
            <mapping type="precomposed">⁊</mapping>
            <mapping type="standardized">et</mapping>
        </char>
        <char xml:id="char211f">
            <desc>RESPONSE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>Rslstrok</value>
            </charProp>
            <mapping type="precomposed">℟</mapping>
            <mapping type="standardized">R</mapping>
        </char>
        <char xml:id="char2184">
            <desc>LATIN ABBREVIATION SIGN SMALL CON</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>conbase</value>
            </charProp>
            <mapping type="precomposed">ↄ</mapping>
            <mapping type="standardized">con</mapping>
        </char>
        <char xml:id="chara751">
            <desc>LATIN SMALL LETTER P WITH STROKE THROUGH DESCENDER</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>pbardes</value>
            </charProp>
            <mapping type="precomposed">ꝑ</mapping>
            <mapping type="standardized">p</mapping>
        </char>
        <char xml:id="chara750">
            <desc>LATIN CAPITAL LETTER P WITH STROKE THROUGH DESCENDER</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>Pbardes</value>
            </charProp>
            <mapping type="precomposed">Ꝑ</mapping>
            <mapping type="standardized">P</mapping>
        </char>
        <char xml:id="chara753">
            <desc>LATIN SMALL LETTER P WITH FLOURISH</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>pflour</value>
            </charProp>
            <mapping type="precomposed">ꝓ</mapping>
            <mapping type="standardized">p</mapping>
        </char>
        <char xml:id="chara759">
            <desc>LATIN SMALL LETTER Q WITH DIAGONAL STROKE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>qslstrok</value>
            </charProp>
            <mapping type="precomposed">ꝙ</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        <char xml:id="chara75d">
            <desc>LATIN SMALL LETTER RUM ROTUNDA</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>rum</value>
            </charProp>
            <mapping type="precomposed">ꝝ</mapping>
            <mapping type="standardized">rum</mapping>
        </char>
        <char xml:id="chara75f">
            <desc>LATIN SMALL LETTER V WITH DIAGONAL STROKE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>vdiagstrok</value>
            </charProp>
            <mapping type="precomposed">ꝟ</mapping>
            <mapping type="standardized">v</mapping>
        </char>
        <char xml:id="chara76d">
            <desc>LATIN SMALL LETTER IS</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>is</value>
            </charProp>
            <mapping type="precomposed">ꝭ</mapping>
            <mapping type="standardized">is</mapping>
        </char>

        <char xml:id="chara770">
            <desc>MODIFIER LETTER US</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>usmod</value>
            </charProp>
            <mapping type="precomposed">ꝰ</mapping>
            <mapping type="standardized">us</mapping>
        </char>

        <char xml:id="chare682">
            <desc>LATIN SMALL LETTER Q WITH DOT ABOVE</desc>
            <charProp>
                <localName>entity</localName>
                <value>qdot</value>
            </charProp>
            <mapping type="MUFI" subtype="PUA">U+E682</mapping>
            <mapping type="composed">q̇</mapping>
            <mapping type="standardized">q</mapping>
            <graphic mimeType="image/png" url="http://v2.manuscriptorium.com/apps/gbank/data/mufi-graphic/e682.png"/>
        </char>

        <char xml:id="chare8b3">
            <desc>LATIN SMALL LETTER Q LIGATED WITH R ROTUNDA</desc>
            <charProp>
                <localName>entity</localName>
                <value>q2app</value>
            </charProp>
            <mapping type="MUFI" subtype="PUA">U+E8B3</mapping>
            <mapping type="standardized">q</mapping>
            <graphic mimeType="image/png" url="http://v2.manuscriptorium.com/apps/gbank/data/mufi-graphic/e8b3.png"/>
        </char>
        <char xml:id="chare8bf">
            <desc>LATIN SMALL LETTER Q LIGATED WITH FINAL ET</desc>
            <charProp>
                <localName>entity</localName>
                <value>q3app</value>
            </charProp>
            <mapping type="MUFI" subtype="PUA">U+E8BF</mapping>
            <mapping type="standardized">q</mapping>
            <graphic mimeType="image/png" url="http://v2.manuscriptorium.com/apps/gbank/data/mufi-graphic/e8bf.png"/>
        </char>
        <char xml:id="chara757">
            <desc>LATIN SMALL LETTER Q WITH STROKE THROUGH DESCENDER</desc>
            <charProp>
               <localName>entity</localName>
               <value>qbardes</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtD">U+A757</mapping>
            <mapping type="precomposed">ꝗ</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        <char xml:id="chare68b">
            <desc>LATIN SMALL LETTER Q WITH STROKE THROUGH DESCENDER AND TILDE</desc>
            <desc n="Combined">LATIN SMALL LETTER Q WITH STROKE THROUGH DESCENDER + COMBINING TILDE</desc>
            <charProp>
               <localName>entity</localName>
               <value>qbardestilde</value>
            </charProp>
            <charProp>
               <localName>combined-entity</localName>
               <value> qbardestilde = qbardes + combtilde </value>
            </charProp>
            <mapping type="MUFI" subtype="PUA">U+E68B</mapping>
            <mapping type="MUFI" subtype="Combined">U+E68B = A757+ 0303</mapping>
            <mapping type="composed">ꝗ̃</mapping>
            <mapping type="standardized">q</mapping>
            <graphic mimeType="image/png" url="http://v2.manuscriptorium.com/apps/gbank/data/mufi-graphic/e68b.png"/>
        </char>
        <char xml:id="char0308">
            <desc>LATIN SMALL LETTER Q WITH STROKE AND COMBINING DIAERESIS</desc>
            <charProp>
                <localName>combined-entity</localName>
                <value> qdiaeresis = qbardes + combdiaeresis </value>
            </charProp>
            <mapping type="composed">ꝗ̈</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        <char xml:id="charebd1">
            <desc>LATIN SMALL LETTER D ROTUNDA WITH DOT ABOVE</desc>
            <desc>LATIN SMALL LETTER D ROTUNDA + COMBINING DOT ABOVE</desc>
            <charProp>
                <localName>entity</localName>
                <value>drotdot</value>
            </charProp>
            <charProp>
                <localName>combined-entity</localName>
                <value> drotdot = drot + combdot </value>
            </charProp>
            <mapping type="MUFI" subtype="PUA">U+EBD1</mapping>
            <mapping type="MUFI" subtype="Combined">U+EBD1 = A77A + 0307</mapping>
            <mapping type="composed">ꝺ̇</mapping>
            <mapping type="standardized">d</mapping>
            <graphic mimeType="image/png" url="http://v2.manuscriptorium.com/apps/gbank/data/mufi-graphic/ebd1.png"/>
        </char>
        <char xml:id="charf159">
            <desc>LATIN ABBREVIATION SIGN SMALL DE</desc>
            <charProp>
                <localName>entity</localName>
                <value>de</value>
            </charProp>
            <mapping type="MUFI" subtype="PUA">U+F159</mapping>
            <mapping type="standardized">d</mapping>
            <graphic mimeType="image/png" url="http://v2.manuscriptorium.com/apps/gbank/data/mufi-graphic/f159.png"/>
        </char>
        <char xml:id="chary0302">
            <desc>LATIN SMALL LETTER Y WITH CIRCUMFLEX</desc>
            <mapping type="precomposed">ŷ</mapping>
            <mapping type="composed">ŷ</mapping>
            <mapping type="standardized">y</mapping>
        </char>

<!-- Combining Characters -->

        <char xml:id="chara0303">
            <desc>LATIN SMALL LETTER A WITH TILDE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>atilde</value>
            </charProp>
            <charProp>
                <unicodeName>combined-entity</unicodeName>
                <value> atilde = a + combtilde </value>
            </charProp>
            <mapping type="precomposed">ã</mapping>
            <mapping type="composed">ã</mapping>
            <mapping type="standardized">a</mapping>
            <graphic mimeType="image/png" url="http://www.manuscriptorium.com/apps/gbank/data/mufi-graphic/00e3.png"/>
        </char>
        <char xml:id="chara0304">
            <desc>LATIN SMALL LETTER A WITH MACRON</desc>
            <charProp>
               <localName>entity</localName>
               <value>amacr</value>
            </charProp>
            <charProp>
               <localName>combined-entity</localName>
               <value> amacr = a + combmacr</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+0101</mapping>
            <mapping type="MUFI" subtype="Combined">U+0101 = 0061 + 0304</mapping>
            <mapping type="precomposed">ā</mapping>
            <mapping type="composed">ā</mapping>
            <mapping type="standardized">a</mapping>
        </char>
        <char xml:id="charb0307">
            <desc>LATIN SMALL LETTER B WITH DOT ABOVE</desc>
            <mapping type="composed">ḃ</mapping>
            <mapping type="standardized">b</mapping>
        </char>
        <char xml:id="charc0303">
            <desc>LATIN SMALL LETTER C WITH TILDE</desc>
            <mapping type="composed">c̃</mapping>
            <mapping type="standardized">c</mapping>
        </char>
        <char xml:id="charc0327">
            <desc>LATIN SMALL LETTER C WITH CEDILLA</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>ccedil</value>
            </charProp>
            <mapping type="precomposed">ç</mapping>
            <mapping type="composed">ç</mapping>
            <mapping type="standardized">c</mapping>
            <graphic mimeType="image/png" url="http://www.manuscriptorium.com/apps/gbank/data/mufi-graphic/00e7.png"/>
        </char>
        <char xml:id="chare0303">
            <desc>LATIN SMALL LETTER E WITH TILDE</desc>
            <mapping type="precomposed">ẽ</mapping>
            <mapping type="composed">ẽ</mapping>
            <mapping type="standardized">e</mapping>
        </char>
        
        <char xml:id="chare0308">
            <desc>LATIN SMALL LETTER E WITH TWO DOTS ABOVE</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>euml</value>
            </charProp>
            <mapping type="precomposed">ë</mapping>
            <mapping type="composed">ë</mapping>
            <mapping type="standardized">e</mapping>
        </char>
        <char xml:id="chare0327">
            <desc>LATIN SMALL LETTER E WITH CEDILLA</desc>
            <mapping type="precomposed">ȩ</mapping>
            <mapping type="composed">ȩ</mapping>
            <mapping type="standardized">e</mapping>
        </char>
        <char xml:id="chare0304">
            <desc>LATIN SMALL LETTER E WITH MACRON</desc>
            <charProp>
               <localName>entity</localName>
               <value>emacr</value>
            </charProp>
            <charProp>
               <localName>combined-entity</localName>
               <value> emacr = e + combmacr</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+0113</mapping>
            <mapping type="MUFI" subtype="Combined">U+0113 = 0065 + 0304</mapping>
            <mapping type="precomposed">ē</mapping>
            <mapping type="composed">ē</mapping>
            <mapping type="standardized">e</mapping>
        </char>
        <char xml:id="chare0328">
            <desc>LATIN SMALL LETTER E WITH OGONEK</desc>
            <charProp>
               <localName>entity</localName>
               <value>eogon</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+0119</mapping>
            <mapping type="precomposed">ę</mapping>
            <mapping type="composed">ę</mapping>
            <mapping type="standardized">e</mapping>
        </char>
        <char xml:id="chare0302">
            <desc>LATIN SMALL LETTER E WITH CIRCUMFLEX</desc>
            <mapping type="precomposed">ê</mapping>
            <mapping type="composed">ê</mapping>
            <mapping type="standardized">e</mapping>
        </char>
        <char xml:id="chare4e9">
            <desc>LATIN SMALL LETTER E WITH CURL</desc>
            <charProp>
                <localName>entity</localName>
                <value>ecurl</value>
            </charProp>
            <charProp>
                <localName>combined-entity</localName>
                <value>ecurl = e + combcurl</value>
            </charProp>
            <mapping type="MUFI" subtype="Combined">U+E4E9 = 0065 + 1DCE</mapping>
            <mapping type="standardized">e</mapping>
        </char>
        
        <char xml:id="chare0300">
            <desc>LATIN SMALL LETTER E WITH GRAVE</desc>
            <mapping type="precomposed">è</mapping>
            <mapping type="composed">è</mapping>
            <mapping type="standardized">e</mapping>
        </char>
        <char xml:id="charg0304">
            <desc>LATIN SMALL LETTER G WITH MACRON</desc>
            <charProp>
                <localName>entity</localName>
                <value>gmacr</value>
            </charProp>
            <mapping type="precomposed">ḡ</mapping>
            <mapping type="standardized">g</mapping>
        </char>
        <char xml:id="charg1E23">
            <desc>LATIN SMALL LETTER H WITH DOT ABOVE</desc>
            <charProp>
                <localName>entity</localName>
                <value>hdot</value>
            </charProp>
            <mapping type="MUFI">1E23</mapping>
            <mapping type="precomposed">ḣ</mapping>
            <mapping type="composed">ḣ</mapping>
            <mapping type="standardized">h</mapping>
        </char>
        <char xml:id="chari0303">
            <desc>LATIN SMALL LETTER I WITH TILDE</desc>
            <mapping type="precomposed">ĩ</mapping>
            <mapping type="composed">ĩ</mapping>
            <mapping type="standardized">i</mapping>
        </char>
        <char xml:id="chari0300">
            <desc>LATIN SMALL LETTER I WITH GRAVE</desc>
            <mapping type="precomposed">ì</mapping>
            <mapping type="standardized">i</mapping>
        </char>
        <char xml:id="chari0301">
            <desc>LATIN SMALL LETTER I WITH ACUTE</desc>
            <mapping type="precomposed">í</mapping>
            <mapping type="standardized">i</mapping>
        </char>
        <char xml:id="chari0302">
            <desc>LATIN SMALL LETTER I WITH CIRCUMFLEX</desc>
            <mapping type="precomposed">î</mapping>
            <mapping type="composed">î</mapping>
            <mapping type="standardized">i</mapping>
        </char>
        <char xml:id="chari0304">
            <desc>LATIN SMALL LETTER I WITH MACRON</desc>
            <charProp>
                <localName>entity</localName>
                <value>imacr</value>
            </charProp>
            <charProp>
                <localName>combined-entity</localName>
                <value>imacr = i + combmacr</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+012B</mapping>
            <mapping type="precomposed">ī</mapping>
            <mapping type="composed">ī</mapping>
            <mapping type="standardized">i</mapping>
        </char>
        
        <char xml:id="charj0303">
            <desc>LATIN SMALL LETTER J WITH TILDE</desc>
            <mapping type="composed">j̃</mapping>
            <mapping type="standardized">j</mapping>
        </char>
        <char xml:id="charj0308">
            <desc>LATIN SMALL LETTER J WITH DIAERESIS</desc>
            <mapping type="composed">j̈</mapping>
            <mapping type="standardized">j</mapping>
        </char>

        <char xml:id="charl0303">
            <desc>LATIN SMALL LETTER L WITH TILDE</desc>
            <mapping type="composed">l̃</mapping>
            <mapping type="standardized">l</mapping>
        </char>
        <char xml:id="charl0309">
            <desc>LATIN SMALL LETTER L WITH HOOK</desc>
            <charProp>
                <localName>entity</localName>
                <value>lhook</value>
            </charProp>
            <mapping type="composed">l̉</mapping>
            <mapping type="standardized">l</mapping>
        </char>
        
        <char xml:id="charm0303">
            <desc>LATIN SMALL LETTER M WITH TILDE</desc>
            <mapping type="composed">m̃</mapping>
            <mapping type="standardized">m</mapping>
        </char>
        <char xml:id="charm0304">
            <desc>LATIN SMALL LETTER M WITH MACRON</desc>
            <mapping type="composed">m̄</mapping>
            <mapping type="standardized">m</mapping>
        </char>
        
        <char xml:id="charn0303">
            <desc>LATIN SMALL LETTER N WITH TILDE</desc>
            <mapping type="precomposed">ñ</mapping>
            <mapping type="composed">ñ</mapping>
            <mapping type="standardized">n</mapping>
        </char>
        <char xml:id="char00D1">
            <desc>LATIN CAPITAL LETTER N WITH TILDE</desc>
            <charProp>
               <localName>entity</localName>
               <value>Ntilde</value>
            </charProp>
            <charProp>
               <localName>combined-entity</localName>
               <value> Ntilde = N + combtilde</value>
            </charProp>
            <mapping type="MUFI" subtype="Lat1Suppl">U+00D1</mapping>
            <mapping type="MUFI" subtype="Combined">U+00D1 = 004E + 0303</mapping>
            <mapping type="precomposed">Ñ</mapping>
            <mapping type="composed">Ñ</mapping>
            <mapping type="standardized">N</mapping>
        </char>
        <char xml:id="charn0304">
            <desc>LATIN SMALL LETTER N WITH MACRON</desc>
            <mapping type="composed">n̄</mapping>
            <mapping type="standardized">n</mapping>
        </char>
        
        <char xml:id="char0254">
            <desc>LATIN SMALL LETTER OPEN O</desc>
            <mapping type="precomposed">ɔ</mapping>
            <mapping type="standardized">o</mapping>
        </char>
        <char xml:id="charo0300">
            <desc>LATIN SMALL LETTER O WITH GRAVE</desc>
            <mapping type="precomposed">ò</mapping>
            <mapping type="composed">ò</mapping>
            <mapping type="standardized">o</mapping>
        </char>
        <char xml:id="charo0302">
            <desc>LATIN SMALL LETTER O WITH CIRCUMFLEX</desc>
            <mapping type="precomposed">ô</mapping>
            <mapping type="composed">ô</mapping>
            <mapping type="standardized">o</mapping>
        </char>
        <char xml:id="charo0303">
            <desc>LATIN SMALL LETTER O WITH TILDE</desc>
            <mapping type="precomposed">õ</mapping>
            <mapping type="composed">õ</mapping>
            <mapping type="standardized">o</mapping>
        </char>
        <char xml:id="charo00F6">
            <desc>LATIN SMALL LETTER O WITH DIAERESIS</desc>
            <mapping type="MUFI">ö</mapping>
            <mapping type="standardized">o</mapping>
        </char>
        <char xml:id="charo0304">
            <desc>LATIN SMALL LETTER O WITH MACRON</desc>
            <charProp>
               <localName>entity</localName>
               <value>omacr</value>
            </charProp>
            <charProp>
               <localName>combined-entity</localName>
               <value> omacr = o + combmacr</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+014D</mapping>
            <mapping type="MUFI" subtype="Combined">U+014D = 006F + 0304</mapping>
            <mapping type="precomposed">ō</mapping>
            <mapping type="composed">ō</mapping>
            <mapping type="standardized">o</mapping>
        </char>
        <char xml:id="charp0301">
            <desc>LATIN SMALL LETTER P WITH ACUTE ACCENT</desc>
            <mapping type="precomposed">ṕ</mapping>
            <mapping type="composed">ṕ</mapping>
            <mapping type="standardized">p</mapping>
        </char>
        <char xml:id="charp0303">
            <desc>LATIN SMALL LETTER P WITH TILDE</desc>
            <mapping type="composed">p̃</mapping>
            <mapping type="standardized">p</mapping>
        </char>
        <char xml:id="charp0307">
            <desc>LATIN SMALL LETTER P WITH DOT ABOVE</desc>
            <mapping type="composed">ṗ</mapping>
            <mapping type="standardized">p</mapping>
        </char>
        <char xml:id="charq0300">
            <desc>LATIN SMALL LETTER Q WITH GRAVE ACCENT</desc>
            <mapping type="composed">q̀</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        <char xml:id="charq0301">
            <desc>LATIN SMALL LETTER Q WITH ACUTE ACCENT</desc>
            <mapping type="composed">q́</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        <char xml:id="charq0303">
            <desc>LATIN SMALL LETTER Q WITH TILDE</desc>
            <mapping type="composed">q̃</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        <char xml:id="charq0304">
            <desc>LATIN SMALL LETTER Q WITH MACRON</desc>
            <charProp>
               <localName>entity</localName>
               <value>qmacr</value>
            </charProp>
            <charProp>
               <localName>combined-entity</localName>
               <value> qmacr = q + combmacr</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+E681</mapping>
            <mapping type="composed">q̄</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        <char xml:id="charq0308">
            <desc>LATIN SMALL LETTER Q WITH TWO DOTS ABOVE</desc>
            <mapping type="composed">q̈</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        <char xml:id="charq0366">
            <desc>LATIN SMALL LETTER Q WITH SMALL O ABOVE</desc>
            <mapping type="composed">qͦ</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        
        <char xml:id="charr0303">
            <desc>LATIN SMALL LETTER R WITH TILDE</desc>
            <mapping type="composed">r̃</mapping>
            <mapping type="standardized">r</mapping>
        </char>
        <char xml:id="charr0304">
            <desc>LATIN SMALL LETTER R WITH MACRON</desc>
            <mapping type="composed">r̄</mapping>
            <mapping type="standardized">r</mapping>
        </char>
        
        <char xml:id="chars0303">
            <desc>LATIN SMALL LETTER S WITH TILDE</desc>
            <mapping type="composed">s̃</mapping>
            <mapping type="standardized">s</mapping>
        </char>
        <char xml:id="chars0308">
            <desc>LATIN SMALL LETTER S WITH DIAERESIS</desc>
            <charProp>
                <localName>combined-entity</localName>
                <value> sdiaeresis = s U+0073 + U+0308</value>
            </charProp>
            <mapping type="composed">s̈</mapping>
            <mapping type="standardized">s</mapping>
        </char>
        <char xml:id="charS0308">
            <desc>LATIN CAPITAL LETTER S WITH DIAERESIS</desc>
            <charProp>
                <localName>combined-entity</localName>
                <value> Sdiaeresis = S U+0053 + U+0308</value>
            </charProp>
            <mapping type="composed">S̈</mapping>
            <mapping type="standardized">S</mapping>
        </char>
        <char xml:id="chart0303">
            <desc>LATIN SMALL LETTER T WITH TILDE</desc>
            <mapping type="composed">t̃</mapping>
            <mapping type="standardized">t</mapping>
        </char>
        
        <char xml:id="charu0303">
            <desc>LATIN SMALL LETTER U WITH TILDE</desc>
            <mapping type="precomposed">ũ</mapping>
            <mapping type="composed">ũ</mapping>
            <mapping type="standardized">u</mapping>
        </char>
        <char xml:id="charu0304">
            <desc>LATIN SMALL LETTER U WITH MACRON</desc>
            <charProp>
               <localName>entity</localName>
               <value>umacr</value>
            </charProp>
            <charProp>
               <localName>combined-entity</localName>
               <value> umacr = u + combmacr</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtA">U+016B</mapping>
            <mapping type="precomposed">ū</mapping>
            <mapping type="composed">ū</mapping>
            <mapping type="standardized">u</mapping>
        </char>
        <char xml:id="charu0302">
            <desc>LATIN SMALL LETTER U WITH CIRCUMFLEX</desc>
            <mapping type="precomposed">û</mapping>
            <mapping type="standardized">u</mapping>
        </char>
        
        <char xml:id="charx0303">
            <desc>LATIN SMALL LETTER X WITH TILDE</desc>
            <mapping type="composed">x̃</mapping>
            <mapping type="standardized">x</mapping>
        </char>
        <char xml:id="chary0304">
            <desc>LATIN SMALL LETTER Y WITH MACRON</desc>
            <charProp>
               <localName>entity</localName>
               <value>ymacr</value>
            </charProp>
            <charProp>
               <localName>combined-entity</localName>
               <value> ymacr = y + combmacr</value>
            </charProp>
            <mapping type="MUFI" subtype="LatExtB">U+0233</mapping>
            <mapping type="MUFI" subtype="Combined">U+0233 = 0079 + 0304</mapping>
            <mapping type="precomposed">ȳ</mapping>
            <mapping type="composed">ȳ</mapping>
            <mapping type="standardized">y</mapping>
        </char>
        
        <!-- Horizontal bars and lines, cross-like symbols -->
        <char xml:id="char2013">
            <desc>EN DASH</desc>
            <mapping type="MUFI" subtype="GenPunct">U+2013</mapping>
            <mapping type="precomposed">–</mapping>
            <mapping type="standardized">–</mapping>
        </char>
        <char xml:id="char2014">
            <desc>EM DASH</desc>
            <mapping type="MUFI" subtype="GenPunct">U+2014</mapping>
            <mapping type="precomposed">—</mapping>
            <mapping type="standardized">—</mapping>
        </char>
        <char xml:id="char2020">
            <desc>DAGGER</desc>
            <charProp>
                <unicodeName>entity</unicodeName>
                <value>dagger</value>
            </charProp>
            <mapping type="precomposed">†</mapping>
            <mapping type="standardized">+</mapping>
        </char>
        <char xml:id="char271d">
            <desc>LATIN CROSS</desc>
            <mapping type="MUFI" subtype="Dingbats">U+271D</mapping>
            <mapping type="precomposed">✝</mapping>
            <mapping type="standardized">+</mapping>
        </char>
        
        <!-- Combining characters where even the base letter is a four-letter codepoint -->
        <char xml:id="char01310323">
            <desc>LATIN TURNED SMALL LETTER I</desc>
            <mapping type="precomposed">ı̣</mapping>
            <mapping type="standardized">i</mapping>
        </char>
        <char xml:id="char21840303">
            <desc>LATIN ABBREVIATION SIGN SMALL CON WITH TILDE</desc>
            <mapping type="composed">ↄ̃</mapping>
            <mapping type="standardized">con</mapping>
        </char>
        <char xml:id="chara7590303">
            <desc>LATIN SMALL LETTER Q WITH DIAGONAL STROKE AND TILDE</desc>
            <mapping type="composed">ꝙ̃</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        <char xml:id="chare8bf0301">
            <desc>LATIN SMALL LETTER Q LIGATED WITH FINAL ET AND WITH ACUTE ACCENT</desc>
            <mapping type="composed">́</mapping>
            <mapping type="standardized">q</mapping>
        </char>
        <char xml:id="chare8bf0308">
            <desc>LATIN SMALL LETTER Q LIGATED WITH FINAL ET AND WITH TWO DOTS ABOVE</desc>
            <mapping type="composed">̈</mapping>
            <mapping type="standardized">q</mapping>
        </char>

        <!-- Other characters, not present in unicode nor MUFI -->
        <char xml:id="char-turnedfl">
            <desc>LATIN TURNED SMALL LIGATURE FL</desc>
            <mapping type="standardized">fl.</mapping>
        </char>
    </tei:charDecl>
  </xsl:variable>

  <!-- ============================================================
       Root template
       ============================================================ -->

  <xsl:template match="/">
    <xsl:if test="/tei:TEI[1]
                        [//tei:revisionDesc/@status/string() = ('g_enriched_approved', 'h_revised')]
                        [not(some $p in //tei:encodingDesc/tei:editorialDecl/tei:p
                             satisfies contains($p/@xml:id, 'AEW'))]
                        [not(some $p in //tei:encodingDesc/tei:editorialDecl/tei:p
                             satisfies contains($p/@xml:id, 'RW'))]
                        [not(tei:text/@type='work_volume')]">
        <xsl:if test="$output-format = 'csv'">
          <xsl:text>"id","doc_id","file_name","ancestor_id","ancestor_type","facs_id","image_info_url","image_jpg_url","lang","is_in_note","contains_abbr","contains_sic","source_sic","source_corr","target_sic","target_corr","nonbreaking_next_line"</xsl:text>
          <xsl:text>&#10;</xsl:text>
        </xsl:if>
        <xsl:apply-templates select="//lb[not(@sameAs)]" mode="line"/>
    </xsl:if>
  </xsl:template>

  <!-- ============================================================
       Per-line template
       ============================================================ -->

  <xsl:template match="lb" mode="line">
    <xsl:variable name="_this-lb" select="."/>
    <xsl:variable name="_next-lb" select="if (ancestor::note) then (ancestor::note//lb[. >> $_this-lb][not(@sameAs)])[1] else following::lb[not(ancestor::note)][not(@sameAs)][1]"/>

    <xsl:variable name="_ancestor" select="
      (ancestor::l       | ancestor::list      |
       ancestor::p       | ancestor::head      |
       ancestor::lg      | ancestor::epigraph  |
       ancestor::note    | ancestor::titlePage |
       ancestor::div)[last()]
    "/>
    <xsl:variable name="ancestor_id"   select="string($_ancestor/@xml:id)"/>
    <xsl:variable name="ancestor_type" select="local-name($_ancestor)"/>
    <xsl:variable name="_this-cell" select="ancestor::cell[1]"/>
    <xsl:variable name="_this-row"  select="ancestor::row[1]"/>

    <xsl:variable name="_line-texts" select="
        if (ancestor::note) 
        then ($_ancestor//text()[. >> $_this-lb]
                [if ($_next-lb) then (. &lt;&lt; $_next-lb) else true()]
                [not($_this-cell and ancestor::cell 
                     and not(ancestor::cell[1] is $_this-cell)
                     and ancestor::row[1] is $_this-row)]) 
        else ($_ancestor//text()[not(ancestor::note)]
                [. >> $_this-lb]
                [if ($_next-lb) then (. &lt;&lt; $_next-lb) else true()]
                [not($_this-cell and ancestor::cell 
                     and not(ancestor::cell[1] is $_this-cell)
                     and ancestor::row[1] is $_this-row)]
              )"/>

    <xsl:variable name="lang" select="
        distinct-values((
            string(ancestor-or-self::*[attribute::xml:lang][1]/@xml:lang),
            for $n in $_line-texts return string(ancestor::*[attribute::xml:lang][1]/@xml:lang)
        ))[. != '']
    "/>
    <xsl:variable name="contains_abbr" select="boolean(ancestor::abbr or (some $n in $_line-texts satisfies $n/ancestor::abbr))"/>
    <xsl:variable name="contains_sic"  select="boolean(ancestor::sic or  (some $n in $_line-texts satisfies $n/ancestor::sic))"/>

    <!-- This lb's position within a choice -->
    <xsl:variable name="_in_abbr"  select="boolean(ancestor::abbr)"/>
    <xsl:variable name="_in_expan" select="boolean(ancestor::expan)"/>
    <xsl:variable name="_in_sic"   select="boolean(ancestor::sic)"/>
    <xsl:variable name="_in_corr"  select="boolean(ancestor::corr)"/>
    <xsl:variable name="_split_choice" select="
      if ($_in_abbr) then ancestor::abbr/parent::choice
      else if ($_in_expan) then ancestor::expan/parent::choice
      else if ($_in_sic) then ancestor::sic/parent::choice
      else if ($_in_corr) then ancestor::corr/parent::choice
      else ()
    "/>

    <xsl:variable name="_sameAs_lb" select="
      if ($_in_abbr and $_split_choice) then $_split_choice/expan//lb[@sameAs = concat('#', $_this-lb/@xml:id)]
      else if ($_in_sic and $_split_choice) then $_split_choice/corr//lb[@sameAs = concat('#', $_this-lb/@xml:id)]
      else if ($_in_corr and $_split_choice) then $_split_choice/sic//lb[@sameAs = concat('#', $_this-lb/@xml:id)]
      else ()
    "/>

    <!-- NEW: detect if the next lb is inside a choice that straddles the line boundary -->
    <xsl:variable name="_next_in_abbr"  select="boolean($_next-lb/ancestor::abbr)"/>
    <xsl:variable name="_next_in_expan" select="boolean($_next-lb/ancestor::expan)"/>
    <xsl:variable name="_next_in_sic"   select="boolean($_next-lb/ancestor::sic)"/>
    <xsl:variable name="_next_in_corr"  select="boolean($_next-lb/ancestor::corr)"/>
    <xsl:variable name="_next_split_choice" select="
      if ($_next_in_abbr) then $_next-lb/ancestor::abbr/parent::choice
      else if ($_next_in_expan) then $_next-lb/ancestor::expan/parent::choice
      else if ($_next_in_sic) then $_next-lb/ancestor::sic/parent::choice
      else if ($_next_in_corr) then $_next-lb/ancestor::corr/parent::choice
      else ()
    "/>
    <!-- Only use next_split if it exists and is different from our own _split_choice -->
    <xsl:variable name="_has_next_split" select="boolean($_next_split_choice and not($_next_split_choice is $_split_choice))"/>

    <!-- sameAs lb within the _next_split_choice (maps the primary lb to the other branch) -->
    <xsl:variable name="_next_sameAs_lb" select="
      if ($_next_in_abbr and $_next_split_choice) then $_next_split_choice/expan//lb[@sameAs = concat('#', $_next-lb/@xml:id)]
      else if ($_next_in_expan and $_next_split_choice) then $_next_split_choice/abbr//lb[@sameAs = concat('#', $_next-lb/@xml:id)]
      else if ($_next_in_sic and $_next_split_choice) then $_next_split_choice/corr//lb[@sameAs = concat('#', $_next-lb/@xml:id)]
      else if ($_next_in_corr and $_next_split_choice) then $_next_split_choice/sic//lb[@sameAs = concat('#', $_next-lb/@xml:id)]
      else ()
    "/>

    <!-- Line nodes: text/g/choice nodes on this line, EXCLUDING _split_choice,
         _next_split_choice and their descendants -->
    <xsl:variable name="_line-nodes">
      <xsl:choose>
        <xsl:when test="$_split_choice">
          <!-- Part 1: nodes between this lb and the split choice -->
          <xsl:copy-of select="
              $_ancestor//node()[. >> $_this-lb]
                                [. &lt;&lt; $_split_choice]
                                [not($_has_next_split and (. is $_next_split_choice or exists(ancestor::* intersect $_next_split_choice)))]
                                [not($_this-cell and ancestor::cell 
                                 and not(ancestor::cell[1] is $_this-cell)
                                 and ancestor::row[1] is $_this-row)]
                                [      self::text()[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])][not(ancestor::choice)][not(ancestor::g)]
                                    or      self::g[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])][not(ancestor::choice)]
                                    or self::choice[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])]
                                ]
            "/>
          <!-- Part 2: nodes strictly after the split choice (not descendants), up to next lb -->
          <xsl:copy-of select="
              $_ancestor//node()[. >> $_split_choice]
                                [not(exists(ancestor::* intersect $_split_choice))]
                                [if ($_next-lb) then (. &lt;&lt; $_next-lb) else true()]
                                [not($_has_next_split and (. is $_next_split_choice or exists(ancestor::* intersect $_next_split_choice)))]
                                [not($_this-cell and ancestor::cell 
                                 and not(ancestor::cell[1] is $_this-cell)
                                 and ancestor::row[1] is $_this-row)]
                                [      self::text()[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])][not(ancestor::choice)][not(ancestor::g)]
                                    or      self::g[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])][not(ancestor::choice)]
                                    or self::choice[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])]
                                ]
            "/>
        </xsl:when>
        <!-- xsl:otherwise>
          <xsl:copy-of select="
              $_ancestor//node()[. >> $_this-lb]
                                [if ($_next-lb) then (. &lt;&lt; $_next-lb) else true()]
                                [not($_has_next_split and (. is $_next_split_choice or exists(ancestor::* intersect $_next_split_choice)))]
                                [      self::text()[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])][not(ancestor::choice)][not(ancestor::g)]
                                    or      self::g[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])][not(ancestor::choice)]
                                    or self::choice[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])]
                                ]
            "/>
        </xsl:otherwise -->
        <xsl:otherwise>
          <xsl:copy-of select="
              $_ancestor//node()[. >> $_this-lb]
                                [if ($_next-lb) then (. &lt;&lt; $_next-lb) else true()]
                                [not($_has_next_split and (. is $_next_split_choice or exists(ancestor::* intersect $_next_split_choice)))]
                                [not($_this-cell and ancestor::cell 
                                     and not(ancestor::cell[1] is $_this-cell)
                                     and ancestor::row[1] is $_this-row)]
                                [      self::text()[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])][not(ancestor::choice)][not(ancestor::g)]
                                    or      self::g[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])][not(ancestor::choice)]
                                    or self::choice[not(ancestor::note[not(. is $_this-lb/ancestor::note[1])])]
                                ]
            "/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>

    <!-- ====== orig-sic (source_sic): abbr + sic side ====== -->
    <xsl:variable name="_orig-sic-text">
      <!-- PREFIX: from _split_choice -->
      <xsl:if test="$_split_choice">
        <xsl:choose>
          <xsl:when test="$_in_abbr">
            <xsl:variable name="_pfx">
              <xsl:apply-templates select="$_split_choice/abbr//node()[. >> $_this-lb]
                                                                       [not(ancestor::g)]
                                                                       [not($_this-cell and ancestor::cell 
                                                                           and not(ancestor::cell[1] is $_this-cell)
                                                                           and ancestor::row[1] is $_this-row)]"
                                           mode="orig-sic"/>
            </xsl:variable>
            <!-- no opening ⦃: abbreviation started on previous line -->
            <xsl:value-of select="normalize-space($_pfx)"/>
            <xsl:text>⦄</xsl:text>
          </xsl:when>
          <xsl:when test="$_in_sic">
            <xsl:apply-templates select="$_split_choice/sic//node()[. >> $_this-lb]
                                                                    [not(ancestor::g)]
                                                                    [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                         mode="orig-sic"/>
          </xsl:when>
          <xsl:when test="$_in_corr and $_sameAs_lb">
            <xsl:apply-templates select="$_split_choice/sic//node()[. >> $_sameAs_lb]
                                                                    [not(ancestor::g)]
                                                                    [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                         mode="orig-sic"/>
          </xsl:when>
          <xsl:when test="$_in_expan and $_sameAs_lb">
            <xsl:apply-templates select="$_split_choice/abbr//node()[. >> $_sameAs_lb]
                                                                    [not(ancestor::g)]
                                                                    [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                          mode="orig-sic"/>
          </xsl:when>
          <xsl:when test="$_in_expan">
            <xsl:apply-templates select="$_split_choice/abbr//node()[. >> $_this-lb]
                                                                    [not(ancestor::g)]
                                                                    [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                          mode="orig-sic"/>
          </xsl:when>
          <xsl:when test="$_in_corr">
            <xsl:apply-templates select="$_split_choice/sic//node()[. >> $_this-lb]
                                                                    [not(ancestor::g)]
                                                                    [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                          mode="orig-sic"/>
          </xsl:when>
        </xsl:choose>
      </xsl:if>
      <!-- MIDDLE -->
      <xsl:apply-templates select="$_line-nodes" mode="orig-sic"/>
      <!-- SUFFIX: tail of _next_split_choice (orig-sic = abbr/sic side) -->
      <xsl:if test="$_has_next_split">
        <xsl:choose>
          <xsl:when test="$_next_in_abbr">
            <xsl:variable name="_sfx">
              <xsl:apply-templates select="$_next_split_choice/abbr//node()[. &lt;&lt; $_next-lb]
                                                                            [not(ancestor::g)]
                                                                            [not($_this-cell and ancestor::cell 
                                                                                and not(ancestor::cell[1] is $_this-cell)
                                                                                and ancestor::row[1] is $_this-row)]"
                                   mode="orig-sic"/>
            </xsl:variable>
            <xsl:text>⦃</xsl:text>
            <xsl:value-of select="normalize-space($_sfx)"/>
            <!-- no closing ⦄: abbreviation continues on next line -->
          </xsl:when>
          <xsl:when test="$_next_in_sic">
            <xsl:apply-templates select="$_next_split_choice/sic//node()[. &lt;&lt; $_next-lb]
                                                                        [not(ancestor::g)]
                                                                        [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                            mode="orig-sic"/>
          </xsl:when>
          <xsl:when test="$_next_in_expan and $_next_sameAs_lb">
            <xsl:apply-templates select="$_next_split_choice/abbr//node()[. &lt;&lt; $_next_sameAs_lb]
                                                                        [not(ancestor::g)]
                                                                        [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                            mode="orig-sic"/>
          </xsl:when>
          <xsl:when test="$_next_in_expan">
            <xsl:apply-templates select="$_next_split_choice/abbr//node()[not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                            mode="orig-sic"/>
          </xsl:when>
          <xsl:when test="$_next_in_corr and $_next_sameAs_lb">
            <xsl:apply-templates select="$_next_split_choice/sic//node()[. &lt;&lt; $_next_sameAs_lb]
                                                                        [not(ancestor::g)]
                                                                        [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                            mode="orig-sic"/>
          </xsl:when>
          <xsl:when test="$_next_in_corr">
            <xsl:apply-templates select="$_next_split_choice/sic//node()[not(ancestor::g)]
                                                                         [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                            mode="orig-sic"/>
          </xsl:when>
        </xsl:choose>
      </xsl:if>
    </xsl:variable>

    <!-- ====== orig-corr (source_corr): abbr + corr side ====== -->
    <xsl:variable name="_orig-corr-text">
      <!-- PREFIX: from _split_choice -->
      <xsl:if test="$_split_choice">
        <xsl:choose>
          <xsl:when test="$_in_abbr">
            <xsl:variable name="_pfx">
              <xsl:apply-templates select="$_split_choice/abbr//node()[. >> $_this-lb]
                                                                       [not(ancestor::g)]
                                                                       [not($_this-cell and ancestor::cell 
                                                                           and not(ancestor::cell[1] is $_this-cell)
                                                                           and ancestor::row[1] is $_this-row)]"
                                           mode="orig-sic"/>
            </xsl:variable>
            <!-- no opening ⦃: abbreviation started on previous line -->
            <xsl:value-of select="normalize-space($_pfx)"/>
            <xsl:text>⦄</xsl:text>
          </xsl:when>
          <xsl:when test="$_in_corr">
            <xsl:apply-templates select="$_split_choice/corr//node()[. >> $_this-lb]
                                                                     [not(ancestor::g)]
                                                                     [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="orig-corr"/>
          </xsl:when>
          <xsl:when test="$_in_sic and $_sameAs_lb">
            <xsl:apply-templates select="$_split_choice/corr//node()[. >> $_sameAs_lb]
                                                                    [not(ancestor::g)]
                                                                    [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="orig-corr"/>
          </xsl:when>
          <xsl:when test="$_in_expan and $_sameAs_lb">
            <xsl:apply-templates select="$_split_choice/abbr//node()[. >> $_sameAs_lb]
                                                                    [not(ancestor::g)]
                                                                    [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="orig-corr"/>
          </xsl:when>
          <xsl:when test="$_in_expan">
            <xsl:apply-templates select="$_split_choice/abbr//node()[. >> $_this-lb]
                                                                     [not(ancestor::g)]
                                                                     [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="orig-corr"/>
          </xsl:when>
          <xsl:when test="$_in_sic">
            <xsl:apply-templates select="$_split_choice/corr//node()[. >> $_this-lb]
                                                                     [not(ancestor::g)]
                                                                     [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="orig-corr"/>
          </xsl:when>
        </xsl:choose>
      </xsl:if>
      <!-- MIDDLE -->
      <xsl:apply-templates select="$_line-nodes" mode="orig-corr"/>
      <!-- SUFFIX: tail of _next_split_choice (orig-sic = abbr/sic side) -->
      <xsl:if test="$_has_next_split">
        <xsl:choose>
          <xsl:when test="$_next_in_abbr">
            <xsl:variable name="_sfx">
              <xsl:apply-templates select="$_next_split_choice/abbr//node()[. &lt;&lt; $_next-lb]
                                                                            [not(ancestor::g)]
                                                                            [not($_this-cell and ancestor::cell 
                                                                                and not(ancestor::cell[1] is $_this-cell)
                                                                                and ancestor::row[1] is $_this-row)]"
                                   mode="orig-sic"/>
            </xsl:variable>
            <xsl:text>⦃</xsl:text>
            <xsl:value-of select="normalize-space($_sfx)"/>
            <!-- no closing ⦄: abbreviation continues on next line -->
          </xsl:when>
          <xsl:when test="$_next_in_corr">
            <xsl:apply-templates select="$_next_split_choice/corr//node()[. &lt;&lt; $_next-lb]
                                                                          [not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="orig-corr"/>
          </xsl:when>
          <xsl:when test="$_next_in_expan and $_next_sameAs_lb">
            <xsl:apply-templates select="$_next_split_choice/abbr//node()[. &lt;&lt; $_next_sameAs_lb]
                                                                          [not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="orig-corr"/>
          </xsl:when>
          <xsl:when test="$_next_in_expan">
            <xsl:apply-templates select="$_next_split_choice/abbr//node()[not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="orig-corr"/>
          </xsl:when>
          <xsl:when test="$_next_in_sic and $_next_sameAs_lb">
            <xsl:apply-templates select="$_next_split_choice/corr//node()[. &lt;&lt; $_next_sameAs_lb]
                                                                          [not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="orig-corr"/>
          </xsl:when>
          <xsl:when test="$_next_in_sic">
            <xsl:apply-templates select="$_next_split_choice/corr//node()[not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="orig-corr"/>
          </xsl:when>
        </xsl:choose>
      </xsl:if>
    </xsl:variable>

    <!-- ====== edit-sic (target_sic): expan + sic side ====== -->
    <xsl:variable name="_edit-sic-text">
      <xsl:if test="$_split_choice">
        <xsl:choose>
          <xsl:when test="$_in_expan">
            <xsl:apply-templates select="$_split_choice/expan//node()[. >> $_this-lb]
                                                                      [not(ancestor::g)]
                                                                      [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
          <xsl:when test="$_in_sic">
            <xsl:apply-templates select="$_split_choice/sic//node()[. >> $_this-lb]
                                                                    [not(ancestor::g)]
                                                                    [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
          <xsl:when test="$_in_abbr and $_sameAs_lb">
            <xsl:apply-templates select="$_split_choice/expan//node()[. >> $_sameAs_lb]
                                                                     [not(ancestor::g)]
                                                                     [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
          <xsl:when test="$_in_corr and $_sameAs_lb">
            <xsl:apply-templates select="$_split_choice/sic//node()[. >> $_sameAs_lb]
                                                                    [not(ancestor::g)]
                                                                    [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
          <xsl:when test="$_in_abbr">
            <xsl:apply-templates select="$_split_choice/expan//node()[. >> $_this-lb]
                                                                     [not(ancestor::g)]
                                                                     [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
          <xsl:when test="$_in_corr">
            <xsl:apply-templates select="$_split_choice/sic//node()[. >> $_this-lb]
                                                                    [not(ancestor::g)]
                                                                    [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
        </xsl:choose>
      </xsl:if>
      <xsl:apply-templates select="$_line-nodes" mode="edit-sic"/>
      <xsl:if test="$_has_next_split">
        <xsl:choose>
          <xsl:when test="$_next_in_abbr and $_next_sameAs_lb">
            <xsl:apply-templates select="$_next_split_choice/expan//node()[. &lt;&lt; $_next_sameAs_lb]
                                                                          [not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
          <xsl:when test="$_next_in_abbr">
            <xsl:apply-templates select="$_next_split_choice/expan//node()[not(ancestor::g)]
                                                                           [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
          <xsl:when test="$_next_in_sic">
            <xsl:apply-templates select="$_next_split_choice/sic//node()[. &lt;&lt; $_next-lb]
                                                                         [not(ancestor::g)]
                                                                         [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
          <xsl:when test="$_next_in_expan">
            <xsl:apply-templates select="$_next_split_choice/expan//node()[. &lt;&lt; $_next-lb]
                                                                           [not(ancestor::g)]
                                                                           [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
          <xsl:when test="$_next_in_corr and $_next_sameAs_lb">
            <xsl:apply-templates select="$_next_split_choice/sic//node()[. &lt;&lt; $_next_sameAs_lb]
                                                                         [not(ancestor::g)]
                                                                         [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
          <xsl:when test="$_next_in_corr">
            <xsl:apply-templates select="$_next_split_choice/sic//node()[not(ancestor::g)]
                                                                        [not($_this-cell and ancestor::cell 
                                                                           and not(ancestor::cell[1] is $_this-cell)
                                                                           and ancestor::row[1] is $_this-row)]"
                                 mode="edit-sic"/>
          </xsl:when>
        </xsl:choose>
      </xsl:if>
    </xsl:variable>

    <!-- ====== edit-corr (target_corr): expan + corr side ====== -->
    <xsl:variable name="_edit-corr-text">
      <xsl:if test="$_split_choice">
        <xsl:choose>
          <xsl:when test="$_in_expan">
            <xsl:apply-templates select="$_split_choice/expan//node()[. >> $_this-lb]
                                                                      [not(ancestor::g)]
                                                                      [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
          <xsl:when test="$_in_corr">
            <xsl:apply-templates select="$_split_choice/corr//node()[. >> $_this-lb]
                                                                     [not(ancestor::g)]
                                                                     [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
          <xsl:when test="$_in_abbr and $_sameAs_lb">
            <xsl:apply-templates select="$_split_choice/expan//node()[. >> $_sameAs_lb]
                                                                      [not(ancestor::g)]
                                                                      [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
          <xsl:when test="$_in_sic and $_sameAs_lb">
            <xsl:apply-templates select="$_split_choice/corr//node()[. >> $_sameAs_lb]
                                                                     [not(ancestor::g)]
                                                                     [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
          <xsl:when test="$_in_abbr">
            <xsl:apply-templates select="$_split_choice/expan//node()[. >> $_this-lb]
                                                                      [not(ancestor::g)]
                                                                      [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
          <xsl:when test="$_in_sic">
            <xsl:apply-templates select="$_split_choice/corr//node()[. >> $_this-lb]
                                                                     [not(ancestor::g)]
                                                                     [not($_this-cell and ancestor::cell 
                                                                         and not(ancestor::cell[1] is $_this-cell)
                                                                         and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
        </xsl:choose>
      </xsl:if>
      <xsl:apply-templates select="$_line-nodes" mode="edit-corr"/>
      <xsl:if test="$_has_next_split">
        <xsl:choose>
          <xsl:when test="$_next_in_abbr and $_next_sameAs_lb">
            <xsl:apply-templates select="$_next_split_choice/expan//node()[. &lt;&lt; $_next_sameAs_lb]
                                                                           [not(ancestor::g)]
                                                                           [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
          <xsl:when test="$_next_in_abbr">
            <xsl:apply-templates select="$_next_split_choice/expan//node()[not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
          <xsl:when test="$_next_in_corr">
            <xsl:apply-templates select="$_next_split_choice/corr//node()[. &lt;&lt; $_next-lb]
                                                                          [not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
          <xsl:when test="$_next_in_expan">
            <xsl:apply-templates select="$_next_split_choice/expan//node()[. &lt;&lt; $_next-lb]
                                                                           [not(ancestor::g)]
                                                                           [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
          <xsl:when test="$_next_in_sic and $_next_sameAs_lb">
            <xsl:apply-templates select="$_next_split_choice/corr//node()[. &lt;&lt; $_next_sameAs_lb]
                                                                          [not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
          <xsl:when test="$_next_in_sic">
            <xsl:apply-templates select="$_next_split_choice/corr//node()[not(ancestor::g)]
                                                                          [not($_this-cell and ancestor::cell 
                                                                             and not(ancestor::cell[1] is $_this-cell)
                                                                             and ancestor::row[1] is $_this-row)]"
                                 mode="edit-corr"/>
          </xsl:when>
        </xsl:choose>
      </xsl:if>
    </xsl:variable>

    <xsl:variable name="source_sic"  select="normalize-space(if (contains($_orig-sic-text, '€')) then substring-before($_orig-sic-text, '€') else $_orig-sic-text)"/>
    <xsl:variable name="source_corr" select="normalize-space(if (contains($_orig-corr-text, '€')) then substring-before($_orig-corr-text, '€') else $_orig-corr-text)"/>
    <xsl:variable name="target_sic"  select="normalize-space(if (contains($_edit-sic-text, '€')) then substring-before($_edit-sic-text, '€') else $_edit-sic-text)"/>
    <xsl:variable name="target_corr" select="normalize-space(if (contains($_edit-corr-text, '€')) then substring-before($_edit-corr-text, '€') else $_edit-corr-text)"/>

    <xsl:variable name="id" select="string(@xml:id)"/>
    <xsl:variable name="doc_id" select="string(/TEI/@xml:id)"/>
    <xsl:variable name="file_name" select="document-uri(/)"/>
    <xsl:variable name="nonbreaking_next_line" select="string($_next-lb[@break eq 'no']/@xml:id)"/>
    <xsl:variable name="is_in_note" select="boolean(ancestor::note)"/>

    <xsl:variable name="_facs_raw">
      <xsl:choose>
        <xsl:when test="preceding::pb[1]/@sameAs">
          <xsl:variable name="_sameAsID" select="substring(preceding::pb[1]/@sameAs, 2)"/>
          <xsl:value-of select="id($_sameAsID)[self::pb]/@facs"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="preceding::pb[1]/@facs"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>
    <xsl:variable name="facs_id" select="
      if (starts-with($_facs_raw, 'facs:'))
      then substring-after($_facs_raw, 'facs:')
      else string($_facs_raw)
    "/>
    <xsl:variable name="image_info_url" select="
      if (matches($facs_id, '^(W[0-9]{4})-([A-Z])-[0-9]{4}$'))
      then replace($facs_id,
                   '^((W[0-9]{4})-([A-Z])-[0-9]{4})$',
                   'https://facs.salamanca.school/iiif/image/$2!$3!$1')
      else if (matches($facs_id, '^(W[0-9]{4})-[0-9]{4}$'))
      then replace($facs_id,
                   '^((W[0-9]{4})-[0-9]{4})$',
                   'https://facs.salamanca.school/iiif/image/$2!$1')
      else ''
    "/>
    <xsl:variable name="image_jpg_url" select="
      if (matches($facs_id, '^(W[0-9]{4})-([A-Z])-[0-9]{4}$'))
      then replace($facs_id,
                   '^((W[0-9]{4})-([A-Z])-[0-9]{4})$',
                   'https://facs.salamanca.school/iiif/image/$2!$3!$1/full/full/0/default.jpg')
      else if (matches($facs_id, '^(W[0-9]{4})-[0-9]{4}$'))
      then replace($facs_id,
                   '^((W[0-9]{4})-[0-9]{4})$',
                   'https://facs.salamanca.school/iiif/image/$2!$1/full/full/0/default.jpg')
      else ''
    "/>



    <xsl:choose>
      <xsl:when test="$output-format = 'csv'">
        <xsl:value-of select="sal:csv-field($id)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($doc_id)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($file_name)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($ancestor_id)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($ancestor_type)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($facs_id)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($image_info_url)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($image_jpg_url)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($lang)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($is_in_note)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($contains_abbr)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($contains_sic)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($source_sic)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($source_corr)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($target_sic)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($target_corr)"/>
        <xsl:text>,</xsl:text>
        <xsl:value-of select="sal:csv-field($nonbreaking_next_line)"/>
        <xsl:text>&#10;</xsl:text>
      </xsl:when>
      <xsl:otherwise>
        <xsl:text>{  "id": "</xsl:text>
        <xsl:value-of select="sal:json-escape($id)"/>
        <xsl:text>", "doc_id": "</xsl:text>
        <xsl:value-of select="sal:json-escape($doc_id)"/>
        <xsl:text>", "file_name": "</xsl:text>
        <xsl:value-of select="sal:json-escape($file_name)"/>
        <xsl:text>", "ancestor_id": "</xsl:text>
        <xsl:value-of select="sal:json-escape($ancestor_id)"/>
        <xsl:text>", "ancestor_type": "</xsl:text>
        <xsl:value-of select="sal:json-escape($ancestor_type)"/>
        <xsl:text>", "facs_id": "</xsl:text>
        <xsl:value-of select="sal:json-escape($facs_id)"/>
        <xsl:text>", "image_info_url": "</xsl:text>
        <xsl:value-of select="sal:json-escape($image_info_url)"/>
        <xsl:text>", "image_jpg_url": "</xsl:text>
        <xsl:value-of select="sal:json-escape($image_jpg_url)"/>
        <xsl:text>", "lang": [</xsl:text>
        <xsl:value-of select="string-join(for $l in $lang return concat('&quot;', sal:json-escape($l), '&quot;'), ', ')"/>
        <xsl:text>]</xsl:text>
        <xsl:text>, "is_in_note": "</xsl:text>
        <xsl:value-of select="$is_in_note"/>
        <xsl:text>", "contains_abbr": "</xsl:text>
        <xsl:value-of select="$contains_abbr"/>
        <xsl:text>", "contains_sic": "</xsl:text>
        <xsl:value-of select="$contains_sic"/>
        <xsl:text>", "source_sic": "</xsl:text>
        <xsl:value-of select="sal:json-escape($source_sic)"/>
        <xsl:text>", "source_corr": "</xsl:text>
        <xsl:value-of select="sal:json-escape($source_corr)"/>
        <xsl:text>", "target_sic": "</xsl:text>
        <xsl:value-of select="sal:json-escape($target_sic)"/>
        <xsl:text>", "target_corr": "</xsl:text>
        <xsl:value-of select="sal:json-escape($target_corr)"/>
        <xsl:text>", "nonbreaking_next_line": "</xsl:text>
        <xsl:value-of select="sal:json-escape($nonbreaking_next_line)"/>
        <xsl:text>"}</xsl:text>
        <xsl:text>&#10;</xsl:text>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <!-- ============================================================
       Elements to eliminate entirely
       ============================================================ -->
  <xsl:template match="cb | pb | note" mode="#all"/>

  <!-- ============================================================
       lb inside line-nodes → sentinel character
       ============================================================ -->
  <xsl:template match="lb" mode="orig-sic orig-corr edit-sic edit-corr">
    <xsl:text>€</xsl:text>
  </xsl:template>

  <!-- ============================================================
       choice / abbr / expan / sic / corr / orig / reg
       ============================================================ -->

  <xsl:template match="choice" mode="orig-sic">
    <xsl:apply-templates select="abbr | sic | orig" mode="orig-sic"/>
  </xsl:template>
  <xsl:template match="abbr" mode="orig-sic">
    <xsl:variable name="_inner">
      <xsl:apply-templates mode="orig-sic"/>
    </xsl:variable>
    <xsl:text>⦃</xsl:text>
    <xsl:value-of select="normalize-space($_inner)"/>
    <xsl:text>⦄</xsl:text>
  </xsl:template>
  <xsl:template match="sic | orig" mode="orig-sic">
    <xsl:apply-templates mode="orig-sic"/>
  </xsl:template>
  <xsl:template match="expan | corr | reg" mode="orig-sic"/>

  <xsl:template match="choice" mode="orig-corr">
    <xsl:apply-templates select="abbr | corr | orig" mode="orig-corr"/>
  </xsl:template>
  <xsl:template match="abbr" mode="orig-corr">
    <xsl:variable name="_inner">
      <xsl:apply-templates mode="orig-corr"/>
    </xsl:variable>
    <xsl:text>⦃</xsl:text>
    <xsl:value-of select="normalize-space($_inner)"/>
    <xsl:text>⦄</xsl:text>
  </xsl:template>
  <xsl:template match="corr | orig" mode="orig-corr">
    <xsl:apply-templates mode="orig-corr"/>
  </xsl:template>
  <xsl:template match="expan | sic | reg" mode="orig-corr"/>

  <xsl:template match="choice" mode="edit-sic">
    <xsl:apply-templates select="expan | sic | orig" mode="edit-sic"/>
  </xsl:template>
  <xsl:template match="expan | sic | orig" mode="edit-sic">
    <xsl:apply-templates mode="edit-sic"/>
  </xsl:template>
  <xsl:template match="abbr | corr | reg" mode="edit-sic"/>

  <xsl:template match="choice" mode="edit-corr">
    <xsl:apply-templates select="expan | corr | orig" mode="edit-corr"/>
  </xsl:template>
  <xsl:template match="expan | corr | orig" mode="edit-corr">
    <xsl:apply-templates mode="edit-corr"/>
  </xsl:template>
  <xsl:template match="abbr | sic | reg" mode="edit-corr"/>

  <!-- ============================================================
       g elements
       ============================================================ -->
  <xsl:template match="g" mode="orig-sic orig-corr edit-sic edit-corr">
    <xsl:choose>
      <xsl:when test="key('chars', substring(@ref,2), $_charDecl)/mapping[@type='precomposed']">
        <xsl:value-of select="key('chars', substring(@ref,2), $_charDecl)/mapping[@type='precomposed']/text()"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:choose>
          <xsl:when test="key('chars', substring(@ref,2), $_charDecl)/mapping[@type='composed']">
            <xsl:value-of select="key('chars', substring(@ref,2), $_charDecl)/mapping[@type='composed']/text()"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:choose>
              <xsl:when test="string-length(string(.)) gt 0">
                <xsl:value-of select="string(.)"/>                
              </xsl:when>
              <xsl:otherwise>
                <xsl:choose>
                  <xsl:when test="key('chars', substring(@ref,2), $_charDecl)/mapping[@type='standardized']">
                    <xsl:value-of select="key('chars', substring(@ref,2), $_charDecl)/mapping[@type='standardized']"/>
                  </xsl:when>
                  <xsl:otherwise>@</xsl:otherwise>
                </xsl:choose>
              </xsl:otherwise>
            </xsl:choose>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <!-- ============================================================
       Text nodes
       ============================================================ -->
  <xsl:template match="text()" mode="orig-sic orig-corr edit-sic edit-corr">
    <xsl:value-of select="string(.)"/>
  </xsl:template>

  <!-- ============================================================
       Identity/fallback for inline elements
       ============================================================ -->
  <xsl:template match="*" mode="orig-sic">
    <xsl:apply-templates mode="orig-sic"/>
  </xsl:template>
  <xsl:template match="*" mode="orig-corr">
    <xsl:apply-templates mode="orig-corr"/>
  </xsl:template>
  <xsl:template match="*" mode="edit-sic">
    <xsl:apply-templates mode="edit-sic"/>
  </xsl:template>
  <xsl:template match="*" mode="edit-corr">
    <xsl:apply-templates mode="edit-corr"/>
  </xsl:template>

  <!-- ============================================================
       Suppress default template in unnamed mode
       ============================================================ -->
  <xsl:template match="text()"/>
  <xsl:template match="*">
    <xsl:apply-templates/>
  </xsl:template>

</xsl:stylesheet>