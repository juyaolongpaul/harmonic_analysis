
\version "2.10.33"

\header {
  title = "55 - Wir Christenleut'"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key b \minor
}

soprano = \relative c {
  \repeat volta 2 {
  \partial 4 b''8 cis 
  d4 cis b } fis' 
  e d cis cis 
  d d e e 
  fis fis e d 
  cis2 b4 fis' 
  e d cis fis 
  e d cis cis 
  d d e e 
  fis fis e d 
  cis2 b4 
}


alto = \relative c {
  \repeat volta 2 {
  \partial 4 fis'4 
  b ais fis } fis8 g 
  a4 fis fis fis 
  fis fis b a 
  a a8 b cis4 fis, 
  fis4. e8 d4 b' 
  b8 ais b4 ais b8 a 
  gis4 a8 b a4 a 
  a8 g fis4 b8 a gis4 
  cis b8 ais b4 b 
  b ais fis
}


tenor = \relative c {
  \repeat volta 2 {
  \partial 4 d'8 e 
  fis4 fis8 e d4 } a8 b 
  cis4 b ais ais 
  b8 cis d4 d cis 
  d d cis b 
  b ais fis d' 
  e fis fis fis 
  b, fis'8 e e4 e 
  fis8 e d cis b4 e 
  e d8 cis b cis d e 
  fis4. e8 dis4
}


baixo = \relative c {
  \repeat volta 2 {
  \partial 4 b4 
  b' fis d } d 
  a' b fis fis 
  b b8 a g e a g 
  fis e d cis b ais b4 
  fis2 b4 b 
  cis d8 e fis4 d 
  e fis8 gis a4 a8 g 
  fis4 b8 a gis4 cis8 b 
  ais4 d gis,8 ais b4 
  fis2 b,4
}


\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}