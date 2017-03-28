\version "2.10.33"

\header {
  title = "69 - Komm, heiliger Geist, Herre Gott"
  composer = "J. S. Bach"
}

global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \partial 4 d4
  e d8 c b c d4
  a2 b4 cis
  d2 r4 d
  e d a8 b c4
  b a g g
  a8 b c4 b8 c d4~ 
  d a g8 fis e4
  d2 r4 d
  g a b8 c d4~ 
  d c b a
  b2 r4 b8 c
  d4 e d a
  b cis d d
  e d a8 b c4
  b a g g
  a8 b c4 b8 c d4~ 
  d a g8 fis e4
  d2 r4 d
  g a b8 c d4~ 
  d c b a
  e fis g a
  b g e2
  d4 a'8 b16 c b2
  a g4
  
}

alto = \relative c'' {
  \partial 4  g4 
  g fis g d8 e 
  fis2 g4 g 
  fis2 r4 g 
  g fis8 e d4 e8 fis 
  g4 fis d d 
  d e d8 e fis4~  
  fis8 g fis e d b cis4 
  a2 r4 a 
  d8 e fis4 g4. a8 
  b4. a8 g fis e4 
  dis2 r4 g 
  fis8 gis a4 a8 g fis e 
  d4 e fis g8 fis 
  e fis g4 fis e 
  d8 e fis4 d e 
  e e8 fis g2~  
  g4 fis g8 d~ d cis
  a2 r4 d8 c 
  b e d c b4. c16 d 
  e4 e e e8 d 
  c2 b4 d 
  d2~  d8 b c4 
  b d2 e4~  
  e d8 c b4
  
}

tenor = \relative c' {
  \partial 4 b4 
  c8 b a4 g g 
  d'2 d4 g,
  a2 r4 g 
  c8 b a g fis g c,4 
  d d'8 c b4 b 
  a g8 a b4 a8 g 
  a4 d d a8 g 
  fis2 r4 fis 
  g c d8 c b a 
  gis fis e4 dis e 
  fis2 r4 g8 a 
  b4 cis d4. c8 
  b a g4 a b 
  c d d g,8 a 
  b g d' c b4 c 
  c8 b a4 b8 a g a 
  b cis d2 a8 g 
  fis2 r4 fis 
  g8 c b a g4. a8 
  b4. a8 g fis e4 
  a2 g4 fis 
  g2. g8 fis 
  g4 a2 g4~  
  g fis16 e fis8 d4
  
}

baixo = \relative c' {
  \partial 4 g4 
  c, d e b8 c 
  d2 g8 fis e4 
  d2 r4 b 
  c d c8 b a4 
  b8 c d4 g, g'
  fis e8 fis g4 fis8 e 
  fis e d cis b g a4 
  d2 r4 d8 c 
  b c b a g4 g'8 f 
  e4 a, b c 
  b2 r4 e 
  b' a8 g fis e d4 
  g8 fis e4 d g 
  c, b8 c d4 e8 fis 
  g c, d4 g, c8 b 
  a4 a' e b8 c 
  d e fis d b' g a4 
  d,2 r4 b 
  e fis g8 a b a 
  gis e a4 e8 d c b 
  c a d4 g, d'
  g,8 a b4 c2 
  g'4 fis g8 fis e d 
  c a d4 g,4
  
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