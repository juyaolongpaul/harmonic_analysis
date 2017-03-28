
\version "2.10.33"

\header {
  title = "98 - Herzlich tut mich verlangen"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key d \major
}

soprano = \relative c {
  \repeat volta 2 {
    \partial 4 fis'4 
    b a g fis 
    e2 fis4 cis' 
    d d cis8 b cis4 
    b2. } d4 
  cis8 b a4 b cis 
  d2 d4 a 
  b a g g 
  fis2. d'4 
  cis8 d e4 d cis 
  b2 cis4 fis, 
  g fis e a 
  fis1 
}


alto = \relative c {
  \repeat volta 2 {
    \partial 4 d'4 
    d d d8 e e d 
    d4 cis d e 
    d8 e fis4 fis fis8 e 
    d2. } b'8 a 
  g4 fis8 e d4 g 
  g fis8 e fis4 fis 
  g fis fis e 
  dis2. e4 
  e e fis e 
  fis e e d 
  d8 cis d4 d cis 
  d1 
}


tenor = \relative c {
  \repeat volta 2 {
    \partial 4 a'4 
    g a b8 a a4 
    b a a ais 
    fis b b ais 
    b2. } fis'4 
  e8 d cis4 b8 a g a 
  b4 a8 g a4 d 
  d d8 cis b4 b 
  b2. b4 
  a a8 gis a b cis a 
  fis b gis4 a a 
  g a b e,8 a 
  a1 
}


baixo = \relative c {
  \repeat volta 2 {
    \partial 4 d4 
    g fis b,8 cis d4 
    g, a d cis 
    b8 cis d e fis4 fis, 
    b2. } b'4 
  e, fis g8 fis e4 
  d2 d4 d 
  g d e8 fis g a 
  b2. gis4 
  a cis, fis8 gis a4 
  d, e a, d8 cis 
  b4 a g a 
  d1 
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