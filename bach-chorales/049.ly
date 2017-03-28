\version "2.10.33"

\header {
  title = "49 - Mit Fried' und Freud' ich fahr' dahin"
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
  b4 a4 b2 
  a2. b4 
  c4 b4 c4 a4   %5
  g8 f8 e4 d4 a'4 
  a4 g4 f8 e8 d4 
  c2 a'4 f4 
  c'4 a4 g8 f8 g4 
  a4 g4 f8 e8 d4   %10
  c4 f4 e8 d4 cis8 
  d1 
  
}

alto = \relative c' {
  \partial 4 a4 
    d4 f4 e4 fis8 gis8 
  a4. gis8 e4 a4~ 
  a8 gis8 a4 f4 e8 d8 
  c2. g'4 
  g4 g4 g4 f4   %5
  e8 d8 cis4 a4 e'8 d8 
  e8 cis8 d e8 a,4 bes4 
  a2 e'4 d4 
  c8 g'4 f8 e8 d8 cis8 d8 
  e4 e4 d8 c8 b4   %10
  a8 b8 cis8 d8 e8 f8 g4~ 
  g4 fis8 e8 fis2 
  
}

tenor = \relative c {
  \partial 4 f8  g8 
    a4 d8 c8 b4 a8 b8 
  c8 d16 e16 f8 e8 c4 e4 
  f8 e16 d16 c16 b16 a4 d16 b16 gis4 
  a2. d4 
  e4 d4 c4. d8   %5
  cis8 d8 a8 g8 f4 a4 
  g8 f4 e8 f2 
  f2 a4 a4 
  g4 c8 d8 e8 f8 e8 d8 
  cis4 a4 a8 g8 f4   %10
  e4 a2. 
  bes4 a8 g8 a2 
  
}

baixo = \relative c {
  \partial 4 d8  e8 
    f8 e8 f8 d8 e8 d8 c8 b8 
  a8 c8 d8 e8 a,4 a'4 
  d,8 e8 f8 e8 d8 b8 e4 
  a,2. g'8 f8 
  e8 f8 g8 f8 e8 c8 f8 d8   %5
  bes'8 g8 a4 d,4 cis8 b8 
  cis8 a8 b8 cis8 d8 c8 d8 e8 
  f2 cis4 d4 
  e4 f4 g8 a8 bes4 
  a4 cis,4 d4 gis,4   %10
  a4. b8 cis8 d8 a4 
  d,1 
  
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