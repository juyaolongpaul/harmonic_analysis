
\version "2.10.33"

\header {
  title = "359 - Allein zu dir, Herr Jesu Christ"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key b \minor
}

soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 d4 
    a b8 cis d e fis4 
    e d2 cis4 
    d2 r4 fis 
    e d cis8 b a4 
    b cis8 d e2~
    e4 d4 cis2 
    b r4 } fis' 
  g e fis d 
  e fis b, fis' 
  g e fis d 
  \time 2/4
  e fis
  \time 4/4
  b,2 r4 d cis b 
  a b g g 
  fis r4 
  r4 b4 
  a d4~ d4 cis4 
  d r4 r4 a4 
  b cis d8 e fis2 e4 e d 
  cis cis 
  b2 r4 
}


alto = \relative c' {
  \repeat volta 2 {
  \partial 4 fis8 g 
    a4 g a ais8 b 
    cis4 fis, e e 
    fis2 r4 a 
    a8 g fis4 g fis 
    g2. fis8 gis 
    ais4 b8 a g4 fis8 e 
    d2 r4 } b' 
  b a a g 
  g fis fis b 
  b a a g8 fis 
  e4 a g2 
  r4 fis eis fis8 gis 
  cis,4 fis fis e 
  dis r4 r4 e2 a4~ a4 g4 
  fis r4 r4 fis4 
  e e d d'8 cis 
  b ais b cis ais4 b8 a 
  g4 fis8 e 
  dis2 r4 
}


tenor = \relative c {
  \repeat volta 2 {
  \partial 4 a'4 
    d d d d 
    cis d8 cis b4 a 
    a2 r4 d 
    cis d e8 d cis4 
    d e ais,8 b cis4~
    cis4 b2 ais4 
    b2 r4 } d 
  d8 b cis4 cis8 a b4 
  b cis d fis 
  e e d d8 cis 
  b4 b b2 
  r4 b b8 a4 gis8 
  fis4 fis g8 a b cis 
  dis4 r4 r4 b8 cis16 d 
  cis4 d8 e fis4 e 
  d r4 r4 d4 
  gis, ais b8 cis d2 cis4~ 
  cis4 b2 ais4 
  b2 r4 
}


baixo = \relative c {
  \repeat volta 2 {
  \partial 4 d8 e 
    fis4 g fis4. gis8 
    ais4 b8 a gis4 a 
    d,2 r4 d' 
    a b e, fis4~
    fis4 e8 d cis b ais cis 
    fis4 g8 fis e4 fis 
    b,2 r4 } b' 
  e, a d, g8 fis 
  e4 ais, b d' 
  d cis cis b8 a 
  g fis e dis e2 
  r4 b cis dis8 eis
  fis4 dis e8 fis g a 
  b4 r4 r4 g2 fis8 g a4 a, 
  d r4 r4 d4 
  d' cis b4. a8 
  g fis g e fis4 g8 fis 
  e4 fis 
  b,2 r4 
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