\version "2.10.33"

\header {
  title = "231 - Die Nacht ist Kommen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \partial 4 g4 
    a4 b4 c2 
  b2 a4 a4 
  gis4 a4 b2 
  a2 r4 c4 
  c4 c8 b8 a2   %5
  d2 r4 b4 
  c4 a4 b4 a4 
  g2 r4 d'4 
  d4 d4 c2 
  a2 r4 c4   %10
  c4 b4 c8 b8 a4 
  a2 r4 d4 
  c4 b4 a2 
  g1 
  
}

alto = \relative c' {
  \partial 4 d4 
    d4 d4 e4 d4 
  d2 e4 d8 c8 
  b4 e4 f4 e8 d8 
  c2 r4 a'4 
  g8 f8 g4. f16 e16 f8 g8   %5
  a2 r4 g4 
  g4 a4. g4 fis8 
  d2 r4 g4 
  a8 d,8 g4. f8 g8 e8 
  f2 r4 g4   %10
  a4. g8 g4 g4 
  fis2 r4 g4~ 
  g8 fis8 g4 g4 fis4 
  d1 
  
}

tenor = \relative c' {
  \partial 4 b4 
    a4 g4 g4 a4 
  g2 g4 f8 e8 
  e4 e8 a8 a4 gis4 
  a2 r4 e'4 
  e8 d8 e4 c8 cis8 d8 e8   %5
  f2 r4 d4 
  e4 d4 d4. c8 
  b2 r4 b4 
  a4 b4 c2 
  c2 r4 e4   %10
  d4 d4 e8 d8 e4 
  d2 r4 d4 
  g,8 a8 b8 e8 a,8 d4 c8 
  b1 
  
}

baixo = \relative c' {
  \partial 4 g4 
    fis4 f4 e4 fis4 
  g2 cis,4 d4~ 
  d4 c4 d4 e4 
  a,2 r4 a'4 
  e4 c4 f8 g8 f8 e8   %5
  d2 r4 g8 f8 
  e4 fis4 g4 d4 
  g,2 r4 g'4 
  fis4 g8 f8 e8 d8 e8 c8 
  f2 r4 e4   %10
  fis4 g4 c,4 cis4 
  d2 r4 b4 
  e4 d8 c8 d4 d,4 
  g1 
  
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