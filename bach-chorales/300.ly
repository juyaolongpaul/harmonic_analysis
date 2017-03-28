\version "2.10.33"

\header {
  title = "300 - Warum betrübst du dich, mein Herz"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \partial 4 a4 
    a8 b8 c4 b4 e4 
  d4 c4 b4 c8 b8 
  a4 e'4 d4 e4 
  a,8 b8 cis4 d4 c4 
  b4 e4 d4 c4   %5
  b2. a8 b8 
  c4 c4 d4 d4 
  e4 e4 c4 e4 
  d4 c4 b4 a8 b8 
  c4 b4 a4   %10
  
}

alto = \relative c' {
  \partial 4 e4 
    e4. a8 a8 gis8 g4 
  f4 fis4 b,4 e4 
  f4 g4 g4 g4 
  f4 e4 f4 e4 
  e4 e8 fis8 gis4 a4   %5
  gis2. a8 gis8 
  a4 a8 g8 f4 bes4 
  bes4 a8 g8 a4 c4 
  b4 a8 g8 fis8 gis8 a4 
  a4 gis4 e4   %10
  
}

tenor = \relative c' {
  \partial 4 c8  d8 
    e4 a,4 b4 bes4 
  a4 a4 gis4 a4 
  a4 c4 c8 b16 a16 b8 cis8 
  d4 a4 a4 a4 
  gis4 a4 b4 c8 d8   %5
  e2. e8 d8 
  c4 ees4 d4 g4 
  g4 f8 e8 f4 g4 
  g8 f8 e4 d4 c8 d8 
  e4. d8 cis4   %10
  
}

baixo = \relative c' {
  \partial 4 a8  b8 
    c4 a4 e4 cis4 
  d4 dis4 e4 a,4 
  d4 c4 g'4 e4 
  f8 g8 a4 d,4 a'4 
  e4 c4 b4 a4   %5
  e'2. c'8 b8 
  a8 g8 f4 bes8 a8 g4 
  c4 c,4 f4 c'4 
  g4 a4 d,8 e8 f4 
  c8 d8 e4 a,4   %10
  
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