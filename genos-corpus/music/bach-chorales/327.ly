\version "2.10.33"

\header {
  title = "327 - Jesu, nun sei gepreiset"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key d \major
}

soprano = \relative c {
  \partial 4 fis''4 
  \repeat volta 2 {
  fis d e fis 
  g2 e4 e 
  e d d cis 
  d2. e4 
  e e e fis 
  d2 d4 d 
  cis d e e 
  d2. }
  \repeat volta 2 {cis4 
  cis cis cis cis 
  d2 d4 } 
  \repeat volta 2 { fis 
  fis d e fis 
  g2 fis4 fis 
  e fis dis dis 
  e2. } fis4 
  fis d e fis 
  g2 e4 e 
  e d d cis 
  d2. 
}


alto = \relative c {
  \partial 4 a''4 
  \repeat volta 2 {
  a fis g a 
  b2 g4 g 
  fis fis e a 
  a2. a4 
  e gis a ais 
  b2 b4 fis 
  e fis g8 a16 b a8 e 
  fis2. }
  \repeat volta 2 { e4 
  a a a a 
  a2 a4 }
  \repeat volta 2 { a 
  a fis g a 
  g a a b 
  g a fis fis 
  g2. } fis4 
  fis fis e dis 
  e fis g g 
  fis fis e e 
  fis2. 
}


tenor = \relative c {
  \partial 4 d'4 
  \repeat volta 2 {
  d a b c 
  d2 c4 c 
  cis d b e 
  fis2. cis4 
  b8 cis d4 cis cis 
  d8 e fis4 fis b, 
  g a8 d d4 cis 
  a2. } 
  \repeat volta 2 { a4 
  e' e fis g8 e 
  fis2 fis4 } 
  \repeat volta 2 { d 
  d a b c 
  d a d fis 
  b, c b b 
  b2. } cis4 
  d fis b, a 
  b2 ais4 b 
  cis b b a 
  a2. 
}


baixo = \relative c {
  \partial 4 d4 
  \repeat volta 2 {
  d' c b a 
  g2 c,4 b' 
  ais b g a 
  d,2. a'4 
  gis e a fis 
  b2 b4 b 
  e, d a' a, 
  d2. }
  \repeat volta 2 { a4 
  a' g fis a 
  d,2 d4 }
  \repeat volta 2 { d 
  d' c b a 
  b cis d dis, 
  e a b b, 
  e2. } ais4 
  b a g fis 
  e d cis b 
  ais b gis a 
  d,2. 
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