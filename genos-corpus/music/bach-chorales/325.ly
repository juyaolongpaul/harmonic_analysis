\version "2.10.33"

\header {
  title = "325 - Mit Fried und Freud ich fahr dahin"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \partial 4 d4 
    a'4 a4 g4 d'4 
  c4 b4 a4 c4 
  a4 c4 b8 a8 b4 
  a2. b4 
  c4 b4 c4 a4   %5
  g4 f4 g4 e4 
  d2 a'4. g8 
  f8 e8 d4 c2 
  a'4 f4 c'4 a4 
  g8 f8 g4 a4 g4   %10
  f4 e8 d8 cis4 f4 
  e8 d8 e4 d4 
  
}

alto = \relative c' {
  \partial 4 a4 
    e'4 f4 e4 d8 e8 
  e8 fis8 gis4 e4 g4 
  f4 f4 b,8 c8 d4 
  cis2. d4 
  e4 f4 g4 f4   %5
  e4 f8 e8 d4 c4 
  b2 e4. cis8 
  d8 c8 bes4 a2 
  e'4 f4 g4 f4 
  e8 d8 cis8 d8 cis4 cis4   %10
  d8 c8 bes4 a4 d4 
  e8 f8 cis4 a4 
  
}

tenor = \relative c {
  \partial 4 f4 
    a4 d4 b4 a8 b8 
  c8 d8 e8 d8 c4 c4~ 
  c8 b8 a4 gis8 a4 gis8 
  a2. f4 
  g4 d'4 c4 c4   %5
  bes4 c4 g4 g4 
  g2 a4 a4 
  a8 g8 f4 f2 
  a4 a4 g4 c4 
  bes8 a8 g4 e4 a4   %10
  a4 g8 f8 e4 a4 
  a4. g8 fis4 
  
}

baixo = \relative c {
  \partial 4 d4 
    cis4 d4 e4 fis8 gis8 
  a4 e4 a,4 e'4 
  f4 e8 d8 e2 
  a,2. d4 
  c4 d4 e4 f4   %5
  g4 a4 b4 c4 
  g2 cis,4. a8 
  d4. e8 f2 
  cis4 d4 e4 f4 
  g8 a8 bes4 a4 a,4   %10
  d4 g,4 g'8 f8 e8 d8 
  cis8 d8 a4 d4 
  
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